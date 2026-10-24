"""Resumable single-user LangGraph runtime; dependencies are supplied explicitly."""

from dataclasses import asdict, replace
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from .audit import AuditLogger
from .config import private_dir
from .context import ContextAssembler, TaskRecord
from .curation import eligible_turn, source_references
from .credentials import ollama_api_key
from .memory_view import render_memories
from .prompts import EXTRACTION_PROMPT, MEMORY_SCHEMA, ANSWER_NOTE_RECOVERY_PROMPT, NOTE_RECOVERY_SCHEMA, system_prompt
from .memory_policy import answer_note_expected, note_excerpts, resolve_note_candidate
from .security import redact
from .tools import ToolRegistry, terminal_approval


class State(TypedDict, total=False):
    messages: list[dict]
    summary: str
    task: dict
    memories: list[dict]
    user_turns: list[dict]
    iterations: int
    manifest: dict
    usage: dict
    workspace: str
    updated_at: str


class AgentRuntime:
    def __init__(self, settings, backend, *, memory=None, approve=terminal_approval,
                 tools=None, on_text=None, on_status=None, on_answer=None, curator=None, max_steps=12, check_cancelled=None):
        self.settings, self.backend = settings, backend
        self.check_cancelled = check_cancelled or (lambda: None)
        self.memory = memory if settings.memory_enabled else None
        self.on_text, self.max_steps = on_text, max_steps
        self.on_status = on_status or (lambda event: None)
        self.on_answer = on_answer or (lambda text: None)
        self.curator = curator
        self.memory_view = None
        self.last_trace, self.warnings = [], []
        private_dir(settings.state_dir)
        self.lock = open(settings.state_dir / "runtime.lock", "a+")
        os.chmod(self.lock.name, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another Harness process is using this state directory") from None
        self.audit = AuditLogger(settings.state_dir / "audit", settings.retention_days)
        self.tools = tools or ToolRegistry(settings, approve=approve, audit=self.audit, memory=self.memory)
        self.tools.memory = self.memory
        self.assembler = ContextAssembler(settings, backend.count_tokens)
        self.conn = sqlite3.connect(settings.state_dir / "sessions.sqlite", check_same_thread=False)
        os.chmod(settings.state_dir / "sessions.sqlite", 0o600)
        self.checkpointer = SqliteSaver(self.conn, serde=JsonPlusSerializer(
            pickle_fallback=False, allowed_msgpack_modules=[], allowed_json_modules=[]))
        self.checkpointer.setup()
        self._retention()
        graph = StateGraph(State)
        graph.add_node("model", self._model)
        graph.add_node("tools", self._tools)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", lambda state: "tools" if state.get("messages") and state["messages"][-1].get("tool_calls") else END)
        graph.add_edge("tools", "model")
        self.graph = graph.compile(checkpointer=self.checkpointer)

    def _retention(self):
        # Prune whole idle sessions, not individual checkpoints needed for recovery.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.settings.retention_days)).isoformat()
        self.conn.execute("CREATE TABLE IF NOT EXISTS session_activity (id TEXT PRIMARY KEY, touched TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS session_metadata "
                          "(id TEXT PRIMARY KEY, name TEXT NOT NULL, first_prompt_seen INTEGER NOT NULL)")
        # Mark pre-existing sessions once; resuming them must not rename historical chats.
        self.conn.execute("INSERT OR IGNORE INTO session_metadata "
                          "SELECT DISTINCT thread_id, 'UNKNOWN', 1 FROM checkpoints")
        stale = [row[0] for row in self.conn.execute("SELECT id FROM session_activity WHERE touched < ?", (cutoff,))]
        for ident in stale:
            self.checkpointer.delete_thread(ident)
            self.conn.execute("DELETE FROM session_activity WHERE id = ?", (ident,))
            self.conn.execute("DELETE FROM session_metadata WHERE id = ?", (ident,))
        self.conn.commit()
        cutoff_epoch = (datetime.now(timezone.utc) - timedelta(days=self.settings.retention_days)).timestamp()
        for artifact in (self.settings.state_dir / "artifacts").glob("*.json"):
            if artifact.stat().st_mtime < cutoff_epoch:
                artifact.unlink()

    def close(self):
        if self.curator:
            self.curator.close()
        self.conn.close()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()

    @staticmethod
    def config(session):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session):
            raise ValueError("Session ID must contain 1-80 letters, digits, underscores or hyphens")
        return {"configurable": {"thread_id": session}, "recursion_limit": 100}

    def snapshot(self, session):
        return dict(self.graph.get_state(self.config(session)).values)

    def forget_memory(self, ident, session):
        """Delete a confirmed selection and clear the current session's active context."""
        state = self.snapshot(session)
        if state.get("workspace") and state["workspace"] != str(self.settings.workspace):
            raise ValueError("This session belongs to a different workspace")
        if self.curator:
            self.curator.invalidate()
        deleted = self.memory.forget(ident) if self.memory else False
        if deleted:
            state.update(messages=[], summary="", user_turns=[], task=asdict(TaskRecord()), memories=[],
                         workspace=str(self.settings.workspace))
            self.graph.update_state(self.config(session), state)
        return "Memory forgotten." if deleted else "Memory not found or disabled."

    def name_session(self, session, prompt):
        from .sessions import session_name
        self.config(session)
        row = self.conn.execute("SELECT first_prompt_seen FROM session_metadata WHERE id = ?", (session,)).fetchone()
        if row and row[0]:
            return
        is_prompt = bool(prompt.strip()) and (not prompt.lstrip().startswith("/") or prompt.startswith("/plan "))
        self.conn.execute("INSERT OR REPLACE INTO session_metadata VALUES (?, ?, ?)",
                          (session, session_name(prompt) if is_prompt else "UNKNOWN", int(is_prompt)))
        self.conn.commit()

    def load_session(self, session):
        state = self.snapshot(session)
        if not state.get("workspace"):
            raise ValueError("Saved session no longer exists")
        workspace = Path(state["workspace"]).resolve(strict=True)
        settings = replace(self.settings, workspace=workspace)
        self.settings = self.tools.settings = self.assembler.settings = settings
        self.tools.memory_edit = None
        self.memory_view = None
        self.last_trace, self.warnings = [], []
        self._touch(session)
        return state

    def delete_session(self, session):
        self.config(session)
        if self.curator:
            self.curator.invalidate()
        self.checkpointer.delete_thread(session)
        self.conn.execute("DELETE FROM session_activity WHERE id = ?", (session,))
        self.conn.execute("DELETE FROM session_metadata WHERE id = ?", (session,))
        self.conn.commit()

    def _touch(self, session):
        self.conn.execute("INSERT OR REPLACE INTO session_activity VALUES (?, ?)",
                          (session, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def _model(self, state):
        self.check_cancelled()
        if state.get("iterations", 0) >= self.max_steps:
            return {"messages": state["messages"] + [{"role": "assistant", "content":
                    "Tool-step limit reached. Review the results before continuing.", "_id": str(uuid.uuid4())}]}
        started = time.perf_counter()
        self.on_status("generation")
        try:
            web_search_configured = bool(ollama_api_key(self.settings))
        except (OSError, ValueError):
            web_search_configured = False
        packet = self.assembler.assemble(system=system_prompt(web_search_configured), messages=state["messages"],
                    tools=self.tools.schemas, task=state.get("task"), summary=state.get("summary", ""),
                    memories=state.get("memories", []))
        packet.manifest["assembly_seconds"] = time.perf_counter() - started
        self.audit.log("context", packet.manifest)
        self.check_cancelled()
        response, usage = self.backend.complete(packet.request, on_text=self.on_text)
        self.on_status("response_end")
        response = redact(response)
        response["_id"] = str(uuid.uuid4())
        calls = response.get("tool_calls", [])
        if len(calls) > 8 or len({call.get("id") for call in calls}) != len(calls) or any(not call.get("id") for call in calls):
            raise ValueError("Model returned invalid or excessive tool calls; no actions executed")
        self.last_trace.append({"type": "assistant", "message": response})
        self.audit.log("model", {"usage": usage, "tool_names": [c["function"]["name"] for c in calls]})
        return {"messages": packet.retained_messages + [response], "summary": packet.summary,
                "manifest": packet.manifest, "usage": usage, "iterations": state.get("iterations", 0) + 1}

    def _tools(self, state):
        messages = list(state["messages"])
        task = deepcopy(state.get("task") or asdict(TaskRecord()))
        for call in messages[-1]["tool_calls"]:
            self.check_cancelled()
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be an object")
                # Completion needs a matching next task and an actual successful result ID.
                if name == "task_complete":
                    pending = task.get("pending", [])
                    valid_ids = [m.get("tool_call_id", "") for m in messages if m["role"] == "tool"
                                 and json.loads(m["content"]).get("status") == "ok"
                                 and json.loads(m["content"]).get("tool") not in {"task_complete", "record_task"}]
                    if (not pending or pending[0] != args.get("task_description")
                            or not any(i and i in args.get("evidence", "") for i in valid_ids)):
                        raise ValueError("Completion must match the next task and cite a successful tool_call_id")
                if name == "record_task":
                    matches = [m for m in messages if m["role"] == "user" and args.get("user_quote")
                               and args["user_quote"] in m.get("content", "")]
                    if not matches:
                        raise ValueError("Task constraints and decisions require an exact user quote")
                self.on_status(f"tool:{name}")
                if name == "update_memory" and getattr(self.tools, "memory_edit", None):
                    self.tools.memory_edit["sources"] = source_references(self.last_trace)
                result = self.tools.execute(name, args)
                if name == "update_memory" and result["status"] == "ok":
                    state["memories"] = []
                if name == "task_complete" and result["status"] == "ok":
                    task.setdefault("completed", []).append(task["pending"].pop(0))
                    task.setdefault("sources", []).append(result["evidence"])
                if name == "record_task" and result["status"] == "ok":
                    field = "constraints" if args["kind"] == "constraint" else "decisions"
                    if args["user_quote"] not in task.setdefault(field, []):
                        task[field].append(args["user_quote"])
                        task.setdefault("sources", []).append(matches[-1]["_id"])
            except (KeyError, ValueError, TypeError) as exc:
                result = {"status": "error", "content": str(exc), "trust": "untrusted-data"}
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result, ensure_ascii=True), "_id": str(uuid.uuid4())})
            self.last_trace.append({"type": "tool", "name": name, "call_id": call["id"], "result": result})
        return {"messages": messages, "task": task, "memories": state.get("memories", [])}

    @staticmethod
    def _recover(messages):
        """Complete interrupted call groups with unknown outcomes; never replay actions."""
        pending = {}
        for message in messages:
            for call in message.get("tool_calls", []):
                pending[call["id"]] = call
            if message["role"] == "tool":
                pending.pop(message["tool_call_id"], None)
        for ident in pending:
            messages.append({"role": "tool", "tool_call_id": ident, "_id": str(uuid.uuid4()),
                             "content": json.dumps({"status": "interrupted", "trust": "untrusted-data",
                                  "content": "Interrupted execution; outcome and side effects are unknown. Inspect before retrying."})})
        return messages

    def _local_reply(self, session, state, user, text, *, clear=False):
        state["messages"] = ([] if clear else state.get("messages", [])) + [user,
            {"role": "assistant", "content": text, "_id": str(uuid.uuid4())}]
        state["workspace"] = str(self.settings.workspace)
        if clear:
            state.update(summary="", user_turns=[], task=asdict(TaskRecord()), memories=[])
        self.graph.update_state(self.config(session), state)
        return text

    def _extract_memories(self, turns, user_id, assistant_reply=""):
        turns = deepcopy(turns[-8:])
        if turns and turns[-1]["id"] == user_id and assistant_reply:
            turns[-1]["assistant_reply"] = redact(assistant_reply)[:4000]
        packet = self.assembler.assemble(system=EXTRACTION_PROMPT, messages=[{
            "role": "user", "content": json.dumps(turns[-8:]), "_id": user_id}])
        packet.request.update(
            response_format={"type": "json_object", "schema": deepcopy(MEMORY_SCHEMA)},
            chat_template_kwargs={"enable_thinking": False}, reasoning_effort="none", temperature=0,
        )
        previous_cap = 0
        note_only = False
        excerpts = []
        for attempt, requested_cap in enumerate((512, 1024), start=1):
            cap = min(self.settings.output_tokens, requested_cap)
            if cap == previous_cap:
                break
            previous_cap = cap
            packet.request["max_tokens"] = cap
            packet.manifest.update(input_tokens=self.backend.count_tokens(packet.request),
                                   reserved_output_tokens=cap, attempt=attempt)
            if packet.manifest["input_tokens"] > self.settings.input_budget:
                raise ValueError("Memory extraction request exceeds context budget")
            self.audit.log("memory_context", packet.manifest)
            response, _ = self.backend.complete(deepcopy(packet.request))
            try:
                if response.get("_finish_reason") == "length":
                    raise ValueError("Extractor reached its output limit")
                data = json.loads(response["content"])
                if (not isinstance(data, dict) or set(data) != {"candidates"}
                        or not isinstance(data["candidates"], list) or len(data["candidates"]) > (1 if note_only else 5)
                        or any(not isinstance(item, dict) for item in data["candidates"])):
                    raise ValueError("Extractor returned an invalid candidates object")
            except (ValueError, TypeError, KeyError):
                self.audit.log("memory_extraction_invalid", {"attempt": attempt,
                               "finish_reason": response.get("_finish_reason"), "output_cap": cap})
                continue
            diagnostics = []
            candidates = data["candidates"]
            if note_only:
                candidates = []
                for index, selection in enumerate(data["candidates"]):
                    candidate, errors = resolve_note_candidate(selection, turns[-1], excerpts)
                    if errors:
                        diagnostics.append({"candidate_index": index, "errors": errors})
                    else:
                        candidates.append(candidate)
            saved = self.memory.save_candidates(candidates, turns[-8:], str(self.settings.workspace),
                sources=source_references(self.last_trace), diagnostics=diagnostics)
            for rejection in diagnostics:
                self.audit.log("memory_candidate_rejected", {"turn_id": user_id, "attempt": attempt, **rejection})
            self.audit.log("memory_selection", {"turn_id": user_id, "candidate_count": len(candidates),
                "accepted_count": len(saved), "rejected_count": len(diagnostics),
                "outcome": "empty_candidates" if not candidates else "accepted" if saved else "all_rejected"})
            self.audit.log("memory_write", {"ids": [record["id"] for record in saved]})
            if answer_note_expected(turns) and not any(r["kind"] == "answer_note" for r in saved):
                if attempt == 1:
                    note_only = True
                    packet.request["messages"][0]["content"] = ANSWER_NOTE_RECOVERY_PROMPT
                    excerpts = note_excerpts(turns[-1].get("assistant_reply", ""))
                    supplied = deepcopy(turns[-8:])
                    supplied[-1]["answer_excerpts"] = excerpts
                    packet.request["messages"][-1]["content"] = json.dumps(supplied)
                    schema = deepcopy(NOTE_RECOVERY_SCHEMA)
                    schema["properties"]["candidates"]["items"]["properties"]["evidence_id"]["enum"] = [e["id"] for e in excerpts]
                    packet.request["response_format"]["schema"] = schema
                    self.audit.log("memory_note_retry", {"turn_id": user_id, "synchronous": True})
                    continue
                raise ValueError("Research answer note was not saved; see memory selection and rejection diagnostics")
            return
        raise ValueError("Extractor did not return complete valid JSON; no new memories were saved")

    def turn(self, text, session="default"):
        self.last_trace, self.warnings = [], []
        self.memory_view = None
        state = self.snapshot(session)
        if state.get("workspace") and state["workspace"] != str(self.settings.workspace):
            raise ValueError("This session belongs to a different workspace; choose a new --session")
        self.name_session(session, text)
        self._touch(session)
        state["messages"] = self._recover(state.get("messages", []))
        user = {"role": "user", "content": redact(text), "_id": str(uuid.uuid4())}
        from .commands import COMMANDS
        slash = re.match(r"^/[a-z]+(?:\s|$)", text.strip())
        if slash and text.split()[0] not in {command.name for command in COMMANDS}:
            return self._local_reply(session, state, user, "Unknown command. Use /help for available commands.")
        memory_on = self.settings.memory_enabled and self.memory is not None
        self.tools.memory_edit = None
        edit = re.fullmatch(r"\s*Modify memory\s+([0-9a-fA-F-]{6,36})\s*:\s*(.*)", text, re.I | re.S)
        if edit:
            if not edit[2].strip():
                return self._local_reply(session, state, user, "Add modification instructions after the memory ID and colon.")
            if not memory_on:
                return self._local_reply(session, state, user, "Memory is disabled for this session.")
            record = self.memory.find(edit[1])
            if not record or record["status"] != "active":
                return self._local_reply(session, state, user, "Active memory not found.")
            if self.curator:
                self.curator.invalidate()
            self.tools.memory_edit = {"id": record["id"], "expected_updated_at": record["updated_at"],
                                      "instruction": user["content"], "turn_id": user["_id"]}
        if text.strip() == "/sessions":
            from .sessions import list_sessions, render_sessions
            return self._local_reply(session, state, user,
                render_sessions(list_sessions(self.settings.state_dir), current=session))
        explicit = re.match(r"^(?:/remember\s+|(?:(?:please|can you|could you|would you)\s+)?remember\s+(?:(?:that|this)\s*:?\s*)?)(.+)$", text, re.I | re.S)
        suffix = re.match(r"^(.+?)[.!]\s*(?:please\s+)?remember (?:this|that)[.!]?$", text, re.I | re.S)
        explicit = (explicit or suffix) if not edit else None
        if explicit:
            if not memory_on:
                return self._local_reply(session, state, user, "Memory is disabled for this session; nothing was saved.")
            fact = explicit.group(1).strip()
            if re.match(r"^(?:what|it|this|that)\b", fact, re.I):
                return self._local_reply(session, state, user, "Please specify the fact to save with /remember FACT.")
            try:
                record = self.memory.save(fact, "explicit", [{"turn_id": user["_id"], "quote": fact}], str(self.settings.workspace))
                reply = f"Remembered: {record['statement']} [memory {record['id']}]"
            except ValueError as exc:
                reply = f"Memory was not saved: {exc}"
            return self._local_reply(session, state, user, reply)
        if text.strip() == "/memories" or text.startswith("/memories "):
            if not memory_on:
                return self._local_reply(session, state, user, "Memory is disabled for this session.")
            argument = text.partition(" ")[2].strip()
            filters = {"saved": "explicit", "facts": "user_stated", "inferred": "inferred", "notes": "answer_note"}
            detail = bool(argument and argument not in filters)
            if detail:
                record = self.memory.find(argument)
                if not record:
                    return self._local_reply(session, state, user, "Memory not found. Use /memories to see saved IDs.")
                records = [record]
            else:
                records = self.memory.records()
                if argument:
                    records = [record for record in records if record["kind"] == filters[argument]]
            self.memory_view = (records, detail)
            return self._local_reply(session, state, user, render_memories(records, detail=detail))
        if text.strip() == "/clear":
            if self.curator:
                self.curator.invalidate()
            return self._local_reply(session, state, user, "Session context cleared.", clear=True)
        state.update(messages=state.get("messages", []) + [user], iterations=0,
                     workspace=str(self.settings.workspace), updated_at=datetime.now(timezone.utc).isoformat())
        if not state.get("task", {}).get("goal"):
            state["task"] = asdict(TaskRecord(goal=user["content"], sources=[user["_id"]]))
        turns = state.get("user_turns", [])
        if eligible_turn(text):
            turns = turns + [{"id": user["_id"], "content": user["content"][:4000]}]
        state["user_turns"] = turns[-8:]
        state["memories"] = []
        if memory_on:
            self.on_status("recall")
            try:
                state["memories"] = self.memory.retrieve(user["content"] + "\n" + state["task"]["goal"], str(self.settings.workspace))
            except Exception as exc:
                self.warnings.append(f"Memory recall unavailable: {exc}")
        self.audit.log("turn", {"session": session, "user_turn_id": user["_id"], "characters": len(text)})
        try:
            result = self.graph.invoke(state, self.config(session))
        finally:
            self.tools.memory_edit = None
        # Deliver the answer before the separate memory-curation generation begins.
        self.on_answer(result["messages"][-1]["content"])
        # Background extraction gets the final answer as context, without raw tool messages.
        if memory_on and eligible_turn(text) and not edit:
            try:
                if self.curator:
                    self.curator.submit(turns, user["_id"], str(self.settings.workspace),
                                        assistant_reply=result["messages"][-1]["content"],
                                        sources=source_references(self.last_trace))
                else:
                    self.on_status("memory")
                    self._extract_memories(turns, user["_id"], result["messages"][-1]["content"])
            except Exception as exc:
                self.warnings.append(f"Automatic memory extraction unavailable: {exc}")
        return result["messages"][-1]["content"]

    def plan(self, goal, session="default"):
        self.name_session(session, goal)
        self.on_status("planning")
        packet = self.assembler.assemble(system="Propose a plan. Return JSON with a tasks array of explicit task strings. Do not execute.",
                                        messages=[{"role": "user", "content": redact(goal), "_id": str(uuid.uuid4())}])
        packet.request["response_format"] = {"type": "json_object"}
        packet.manifest["input_tokens"] = self.backend.count_tokens(packet.request)
        if packet.manifest["input_tokens"] > self.settings.input_budget:
            raise ValueError("Plan request exceeds context budget")
        self.audit.log("plan_context", packet.manifest)
        response, _ = self.backend.complete(packet.request, on_text=lambda text: None)
        tasks = json.loads(response["content"])["tasks"]
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= 30 or any(not isinstance(t, str) or not t.strip() or len(t) > 500 for t in tasks):
            raise ValueError("Model returned an invalid task list")
        request = json.dumps({"goal": redact(goal), "tasks": tasks}, ensure_ascii=True)
        if not self.tools._approve(request, self.settings.workspace, "plan"):
            return "Plan declined; no tasks executed."
        state = self.snapshot(session)
        state["task"] = asdict(TaskRecord(goal=redact(goal), pending=tasks))
        self.graph.update_state(self.config(session), state)
        return self.turn("Execute this approved plan: " + request, session)
