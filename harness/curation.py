"""Durable bounded background curation through a separate Ollama model."""

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import replace
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
import tempfile
import threading
import time

import httpx

from .config import private_dir
from .memory import fact_key
from .prompts import EXTRACTION_PROMPT, MEMORY_SCHEMA, ANSWER_NOTE_RECOVERY_PROMPT, NOTE_RECOVERY_SCHEMA
from .memory_policy import candidate_field_errors, answer_note_expected, note_excerpts, resolve_note_candidate
from .security import contains_secret, redact


def valid_candidate(candidate):
    return not candidate_field_errors(candidate)


def source_references(trace):
    """Capture related URLs locally; no raw tool messages are sent to the extractor."""
    sources = {}
    for event in trace:
        result = event.get("result", {})
        if event.get("type") != "tool" or result.get("status") != "ok":
            continue
        if event.get("name") == "perform_web_search":
            try:
                entries = json.loads(result.get("content", "")).get("results", [])[:5]
            except (ValueError, AttributeError, TypeError):
                continue
        elif event.get("name") == "fetch_url":
            entries = [{"url": result.get("source", "")}]
        else:
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url", "")
            if not isinstance(url, str) or len(url) > 2000 or contains_secret(url):
                continue
            try:
                parsed = urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                    continue
            except ValueError:
                continue
            sources[url] = {"url": url, "captured_at": result.get("source_captured_at", ""),
                            "artifact_id": result.get("artifact_id", "")}
    return list(sources.values())[:5]


def eligible_turn(text):
    return bool(text.strip()) and not contains_secret(text) and "```" not in text and not text.lstrip().startswith(">")


def extraction_turns(turns):
    eligible = [turn for turn in turns[-8:] if eligible_turn(turn["content"])]
    if not eligible:
        return []
    latest = {"id": eligible[-1]["id"], "content": eligible[-1]["content"]}
    if eligible[-1].get("assistant_reply"):
        latest["assistant_reply"] = redact(str(eligible[-1]["assistant_reply"]))[:4000]
    if re.fullmatch(r"\s*(?:hi|hello|hey|thanks|thank you|ok|okay)[.!\s]*", latest["content"], re.I):
        return []
    if not fact_key(latest["content"]).startswith("statement."):
        return [latest]
    stopwords = {"the", "and", "that", "this", "with", "for", "you", "your", "have", "has",
                 "are", "was", "were", "but", "not", "from", "what", "how", "can", "please", "just"}

    def words(text):
        return {word for word in re.findall(r"\w+", text.lower()) if len(word) > 2 and word not in stopwords}

    current_words = words(latest["content"])
    ranked = [(len(current_words & words(turn["content"])), index, turn)
              for index, turn in enumerate(eligible[:-1])]
    prior = []
    if ranked:
        score, _, related = max(ranked, key=lambda item: item[:2])
        if score >= 2:
            prior = [{"id": related["id"], "content": related["content"][:500]}]
    return prior + [latest]


def write_pending(path, jobs):
    descriptor, temporary = tempfile.mkstemp(prefix=".curation-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(jobs, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def discard_pending(state_dir):
    path = state_dir / "curation-pending.json"
    if path.exists():
        write_pending(path, [])


class OllamaExtractor:
    def __init__(self, settings, *, client=None):
        self.settings = settings
        self.client = client or httpx.Client(base_url=settings.memory_server_url,
            timeout=httpx.Timeout(settings.memory_timeout, connect=5), trust_env=False)
        self._model_checked = False
        self._fallback = None
        self._local_only = False
        self._retry_primary_at = 0.0
        self.last_model = None
        self.cloud = settings.memory_model.endswith("-cloud") or settings.memory_model.endswith(":cloud")

    def close(self):
        self.client.close()

    def extract(self, turns, stopped, audit=None, *, note_only=False, feedback=()):
        # No network request or fallback for greetings, excluded turns, or stale jobs.
        if stopped() or not extraction_turns(turns):
            return [], []
        fallback_model = self.settings.memory_fallback_model
        use_fallback = bool(self.cloud and fallback_model and fallback_model != self.settings.memory_model)
        if not use_fallback or time.monotonic() >= self._retry_primary_at:
            try:
                result = self._extract(turns, stopped, audit, note_only=note_only, feedback=feedback)
                self.last_model = self.settings.memory_model
                self._retry_primary_at = 0.0
                return result
            except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
                if stopped():
                    return [], []
                # Metadata can identify a remote model whose alias lacks a cloud suffix.
                if not (self.cloud and fallback_model and fallback_model != self.settings.memory_model):
                    raise
                self._retry_primary_at = time.monotonic() + self.settings.memory_cloud_retry_interval
                if audit:
                    audit.log("memory_fallback", {"primary_model": self.settings.memory_model,
                        "fallback_model": fallback_model, "reason": type(exc).__name__,
                        "phase": "answer_note_recovery" if note_only else "extraction",
                        "retry_primary_after_seconds": self.settings.memory_cloud_retry_interval})
        if stopped():
            return [], []
        if self._fallback is None:
            local = replace(self.settings, memory_model=fallback_model, memory_fallback_model="")
            self._fallback = OllamaExtractor(local, client=self.client)
            self._fallback._local_only = True
        try:
            result = self._fallback._extract(turns, stopped, audit, note_only=note_only, feedback=feedback)
            self.last_model = fallback_model
            return result
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            if stopped():
                return [], []
            message = f"Cloud curation unavailable; local fallback {fallback_model} failed: {exc}"
            if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                raise TimeoutError(message) from exc
            raise ValueError(message) from exc

    def recover_note(self, turns, stopped, audit=None, *, feedback=()):
        return self.extract(turns, stopped, audit, note_only=True, feedback=feedback)

    def _extract(self, turns, stopped, audit=None, *, note_only=False, feedback=()):
        turns = extraction_turns(turns)
        if not turns:
            return [], []
        if not self._model_checked:
            info = self.client.post("/api/show", json={"model": self.settings.memory_model})
            if info.status_code == 404:
                raise ValueError(f"Register the curation model: ollama pull {self.settings.memory_model}; sign in to Ollama for cloud models")
            info.raise_for_status()
            metadata = info.json()
            self.cloud = self.cloud or bool(metadata.get("remote_host") or metadata.get("remote_model"))
            self._model_checked = True
        if self._local_only and self.cloud:
            raise ValueError("The fallback model resolves to a cloud model; pull a local fallback model")
        instructions = (EXTRACTION_PROMPT + "\nEach object's content is a USER turn; its id is an opaque identifier. "
            "An optional assistant_reply field is the final assistant answer for that user turn, supplied only as context. "
            "Only answer_note candidates may quote assistant_reply; never turn assistant guesses into user facts. "
            "Copy that exact id into evidence.turn_id. Never shorten or renumber it. "
            "A turn id is not a fact key. Use user.location, user.name, user.answer_length, or "
            "user.programming_language for those attributes. "
            "Statements such as 'I live in Pune' and 'I prefer concise answers' are durable direct facts, "
            "not one-off tasks. Include clear direct facts as user_stated candidates. "
            "A greeting alone has no durable facts.\nUse this JSON schema:\n")
        if note_only:
            instructions = (ANSWER_NOTE_RECOVERY_PROMPT +
                "\nEach object's content is a user turn; assistant_reply is its final answer. "
                "answer_excerpts contains allowed evidence IDs and exact source slices. "
                "Use this JSON schema:\nPrior rejection codes: " + ", ".join(feedback)[:200] + "\n")
        # Ollama does not expose a preflight tokenizer for this adapter. Use a small
        # UTF-8 byte ceiling plus template margin, never the main model's tokenizer.
        budget = self.settings.memory_context - 1024 - 1024 - 384
        omitted = []
        while True:
            schema = deepcopy(NOTE_RECOVERY_SCHEMA if note_only else MEMORY_SCHEMA)
            supplied = deepcopy(turns)
            excerpts = []
            if note_only:
                excerpts = note_excerpts(turns[-1].get("assistant_reply", ""))
                supplied[-1]["answer_excerpts"] = excerpts
                schema["properties"]["candidates"]["items"]["properties"]["evidence_id"]["enum"] = [e["id"] for e in excerpts]
            else:
                evidence_schema = schema["properties"]["candidates"]["items"]["properties"]["evidence"]["items"]
                evidence_schema["properties"]["turn_id"]["enum"] = [turn["id"] for turn in turns]
            system = instructions + json.dumps(schema)
            content = json.dumps(supplied, ensure_ascii=False)
            size = len((system + content).encode("utf-8"))
            if size <= budget:
                break
            if len(turns) > 1:
                omitted.append(turns.pop(0)["id"])
            elif turns[0].get("assistant_reply"):
                reply = turns[0]["assistant_reply"]
                turns[0]["assistant_reply"] = reply[:len(reply) // 2] if len(reply) > 32 else ""
            elif len(turns[0]["content"]) > 32:
                turns[0]["content"] = turns[0]["content"][:len(turns[0]["content"]) // 2]
            else:
                raise ValueError("Curation prompt exceeds its separate context budget")
        deadline = time.monotonic() + self.settings.memory_timeout
        failure = "unknown"
        previous_fields = ""
        caps = (1024,) if note_only else (512, 1024)
        for attempt, cap in enumerate(caps, 1):
            field_failures = []
            if stopped():
                return [], turns
            if audit:
                audit.log("memory_context", {"model": self.settings.memory_model, "turn_id": turns[-1]["id"],
                    "budget_kind": "utf8-byte-ceiling-with-template-margin", "prompt_bytes": size,
                    "byte_ceiling": budget, "context": self.settings.memory_context,
                    "output_cap": cap, "omitted_turn_ids": omitted, "cloud": self.cloud,
                    "phase": "answer_note_recovery" if note_only else "extraction",
                    "assistant_reply_chars": len(turns[-1].get("assistant_reply", "")),
                    "keep_alive": self.settings.memory_keep_alive})
            retry = ("" if attempt == 1 else
                "\nPrevious response failed validation: " + failure + previous_fields +
                ". Return exactly one complete JSON object with candidates. At most ONE candidate; "
                "keep statement under 300 characters and evidence quotes under 200 characters. "
                "Copy exact turn IDs and quotes. No prose or Markdown. Do not invent missing evidence.")
            request = {"model": self.settings.memory_model, "messages": [
                {"role": "system", "content": system + retry}, {"role": "user", "content": content}],
                "stream": True, "keep_alive": self.settings.memory_keep_alive,
                "options": {"num_predict": cap, "temperature": 0}}
            if self.cloud:
                request["think"] = "low"
            else:
                request["format"] = schema
                request["options"].update(num_gpu=0, num_thread=self.settings.memory_threads,
                                          num_ctx=self.settings.memory_context)
            chunks, result = [], {}
            length = 0
            with self.client.stream("POST", "/api/chat", json=request) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if stopped():
                        return [], turns
                    if time.monotonic() > deadline:
                        raise TimeoutError("Background curation exceeded its deadline")
                    if not line:
                        continue
                    result = json.loads(line)
                    if result.get("error"):
                        raise ValueError(result["error"])
                    text = result.get("message", {}).get("content", "")
                    length += len(text)
                    if length > 131072:
                        raise ValueError("Curation response exceeded its size limit")
                    chunks.append(text)
            if audit:
                audit.log("memory_generation", {"model": self.settings.memory_model, "turn_id": turns[-1]["id"],
                    "prompt_tokens": result.get("prompt_eval_count"), "output_tokens": result.get("eval_count"),
                    "finish_reason": result.get("done_reason"), "cloud": self.cloud,
                    "load_duration_ns": result.get("load_duration"),
                    "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
                    "eval_duration_ns": result.get("eval_duration"),
                    "total_duration_ns": result.get("total_duration")})
            try:
                if not result.get("done"):
                    failure = "incomplete_stream"
                elif result.get("done_reason") == "length":
                    failure = "output_limit"
                else:
                    raw = "".join(chunks).strip()
                    fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", raw, re.I)
                    if fenced:
                        raw = fenced[1]
                    data = json.loads(raw)
                    if not isinstance(data, dict) or set(data) != {"candidates"}:
                        failure = "invalid_envelope"
                    elif not isinstance(data["candidates"], list) or len(data["candidates"]) > (1 if note_only else 5):
                        failure = "invalid_candidates_list"
                    elif note_only:
                        accepted = []
                        for index, selection in enumerate(data["candidates"]):
                            candidate, errors = resolve_note_candidate(selection, turns[-1], excerpts)
                            if errors:
                                field_failures.append({"candidate_index": index, "errors": errors})
                            else:
                                accepted.append(candidate)
                        if accepted:
                            return accepted[:1], turns
                        failure = "answer_note_rejected" if field_failures else "missing_answer_note"
                    elif not all(valid_candidate(candidate) for candidate in data["candidates"]):
                        failure = "invalid_candidate_fields"
                        field_failures = [{"candidate_index": index, "errors": candidate_field_errors(candidate)}
                                          for index, candidate in enumerate(data["candidates"])
                                          if not valid_candidate(candidate)]
                    else:
                        return data["candidates"], turns
            except json.JSONDecodeError:
                failure = "invalid_json"
            except (ValueError, TypeError):
                failure = "invalid_response"
            previous_fields = " (" + ", ".join(sorted({error["field"] for item in field_failures
                for error in item["errors"]}))[:80] + ")" if field_failures else ""
            if audit:
                audit.log("memory_extraction_invalid", {"attempt": attempt, "output_cap": cap,
                    "reason": failure, "candidate_errors": field_failures, "model": self.settings.memory_model,
                    "turn_id": turns[-1]["id"],
                    "phase": "answer_note_recovery" if note_only else "extraction",
                    "response_chars": length, "finish_reason": result.get("done_reason")})
        if note_only:
            raise ValueError("Answer-note recovery failed (" + failure + "); no answer note was saved")
        raise ValueError("Curation failed validation after two attempts (" + failure +
                         "); no new memories were saved")



class MemoryCurator:
    def __init__(self, settings, memory, *, audit=None, on_activity=None, extractor=None):
        self.settings, self.memory, self.audit = settings, memory, audit
        self.on_activity = on_activity or (lambda message: None)
        self.extractor = extractor or OllamaExtractor(settings)
        self._condition = threading.Condition()
        self._stopped = False
        self._epoch = 0
        self._active = None
        self._active_epoch = 0
        self._omitted = 0
        self._pending = OrderedDict()
        self.path = private_dir(settings.state_dir) / "curation-pending.json"
        if self.path.exists():
            try:
                if self.path.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError("Pending memory queue exceeds its size limit")
                jobs = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(jobs, list):
                    raise ValueError("Pending memory queue must be a list")
                cutoff = time.time() - settings.retention_days * 86400
                for job in jobs[-32:]:
                    if job["created"] >= cutoff:
                        self._pending[job["id"]] = job
            except (OSError, ValueError, KeyError, TypeError):
                self._pending.clear()
                self.on_activity("Pending memory queue could not be read; no queued facts were loaded")
        self._thread = threading.Thread(target=self._work, name="harness-memory-curation", daemon=True)
        self._thread.start()

    def _persist(self):
        active = [self._active] if self._active and self._active_epoch == self._epoch else []
        jobs = active + list(self._pending.values())
        write_pending(self.path, jobs)

    def submit(self, turns, user_id, workspace, *, assistant_reply="", sources=None):
        job = {"id": user_id, "turns": deepcopy(turns[-8:]), "workspace": workspace, "created": time.time()}
        job["sources"] = deepcopy(sources or [])[:5]
        if job["turns"] and job["turns"][-1]["id"] == user_id and assistant_reply:
            job["turns"][-1]["assistant_reply"] = redact(assistant_reply)[:4000]
        with self._condition:
            if self._stopped:
                return
            if len(self._pending) >= 32 - bool(self._active):
                dropped, _ = self._pending.popitem(last=False)
                self._omitted += 1
                if self.audit:
                    self.audit.log("memory_queue_omission", {"turn_id": dropped, "reason": "queue capacity"})
                self.on_activity("Memory queue full: oldest automatic update skipped; /remember saves immediately")
            self._pending[user_id] = job
            self._persist()
            suffix = f" ({self._omitted} old queued updates omitted)" if self._omitted else ""
            self.on_activity("Curating memory in background…" + suffix)
            self._condition.notify_all()

    def invalidate(self):
        with self._condition:
            self._epoch += 1
            self._pending.clear()
            self._persist()
            self.on_activity("")

    def _current(self, epoch):
        with self._condition:
            return not self._stopped and epoch == self._epoch

    def _save_result(self, job, epoch, candidates, turns, phase):
        diagnostics = []
        records = self.memory.save_candidates(candidates, turns, job["workspace"],
            sources=job.get("sources", []),
            observed_at=datetime.fromtimestamp(job["created"], timezone.utc).isoformat(),
            before_commit=lambda: self._current(epoch), diagnostics=diagnostics)
        if self.audit:
            metadata = {"turn_id": job["id"], "phase": phase,
                        "model": getattr(self.extractor, "last_model", None) or self.settings.memory_model}
            for rejection in diagnostics:
                self.audit.log("memory_candidate_rejected", {**metadata, **rejection})
            self.audit.log("memory_selection", {**metadata, "candidate_count": len(candidates),
                "accepted_count": len(records), "rejected_count": len(diagnostics),
                "outcome": "empty_candidates" if not candidates else "accepted" if records else "all_rejected"})
            self.audit.log("memory_write", {**metadata, "ids": [record["id"] for record in records], "background": True})
        return records, diagnostics

    def _curate(self, job, epoch):
        def stopped():
            return not self._current(epoch)
        candidates, turns = self.extractor.extract(job["turns"], stopped, self.audit)
        if stopped():
            return
        records, diagnostics = self._save_result(job, epoch, candidates, turns, "extraction")
        if (stopped() or any(record["kind"] == "answer_note" for record in records)
                or not answer_note_expected(job["turns"])):
            return
        feedback = sorted({error["reason"] for entry in diagnostics for error in entry["errors"]})
        if not feedback:
            feedback = ["missing_answer_note"]
        if self.audit:
            self.audit.log("memory_note_retry", {"turn_id": job["id"], "reasons": feedback,
                "accepted_fact_count": len(records)})
        recover = getattr(self.extractor, "recover_note", None)
        if recover is None:
            raise ValueError("Research answer note was not saved; the extractor does not support recovery")
        try:
            candidates, turns = recover(job["turns"], stopped, self.audit, feedback=feedback)
            if stopped():
                return
            records, _ = self._save_result(job, epoch, candidates, turns, "answer_note_recovery")
            if not any(record["kind"] == "answer_note" for record in records):
                raise ValueError("No answer note passed evidence and acceptance checks")
        except Exception as exc:
            if stopped():
                return
            message = "Research answer note was not saved; other accepted memories remain: " + str(exc)
            if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                raise TimeoutError(message) from exc
            raise ValueError(message) from exc

    def _work(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopped or self._pending)
                    if self._stopped:
                        return
                    _, job = self._pending.popitem(last=False)
                    self._active = job
                    epoch = self._epoch
                    self._active_epoch = epoch
                    self._persist()
                failed = False
                try:
                    self._curate(job, epoch)
                except Exception as exc:
                    failed = True
                    if self._current(epoch):
                        timed_out = isinstance(exc, (httpx.TimeoutException, TimeoutError))
                        if timed_out and job.get("timeout_retries", 0) < 1:
                            with self._condition:
                                if self._stopped or epoch != self._epoch:
                                    continue
                                job["timeout_retries"] = 1
                                self._pending[job["id"]] = job
                            message = "Memory curation is taking longer; retrying once in background…"
                        else:
                            message = "Memory curation unavailable: " + str(redact(str(exc)))
                        if self.audit:
                            self.audit.log("memory_error", {"message": message})
                        self.on_activity(message)
                finally:
                    with self._condition:
                        if self._stopped:
                            return
                        self._active = None
                        self._persist()
                        if not failed:
                            self.on_activity("Curating memory in background…" if self._pending else "")
                        self._condition.notify_all()
        finally:
            self.extractor.close()

    def wait_idle(self, timeout=30):
        with self._condition:
            return self._condition.wait_for(lambda: not self._pending and self._active is None, timeout)

    def close(self):
        with self._condition:
            self._stopped = True
            self._persist()
            self._condition.notify_all()
        # In-flight HTTP generation stops at the next streamed chunk. Its job is
        # durable; no late database commit is allowed after this method returns.
        self._thread.join(timeout=.2)
