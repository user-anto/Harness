"""Durable bounded background curation through a separate Ollama model."""

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
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
from .prompts import EXTRACTION_PROMPT, MEMORY_SCHEMA
from .security import contains_secret, redact


def valid_candidate(candidate):
    if not isinstance(candidate, dict) or set(candidate) != set(MEMORY_SCHEMA["properties"]["candidates"]["items"]["required"]):
        return False
    if (not isinstance(candidate["statement"], str) or not 1 <= len(candidate["statement"]) <= 1000
            or not isinstance(candidate["fact_key"], str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{3,120}", candidate["fact_key"])
            or candidate["kind"] not in ("user_stated", "inferred", "answer_note")
            or type(candidate["confidence"]) not in (int, float) or not 0 <= candidate["confidence"] <= 1
            or not isinstance(candidate["evidence"], list) or not 1 <= len(candidate["evidence"]) <= 8):
        return False
    return all(isinstance(item, dict) and set(item) == {"turn_id", "quote"}
               and all(isinstance(item[key], str) and item[key] for key in ("turn_id", "quote"))
               and len(item["quote"]) <= 1000 for item in candidate["evidence"])


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
        self.cloud = settings.memory_model.endswith("-cloud") or settings.memory_model.endswith(":cloud")

    def close(self):
        self.client.close()

    def extract(self, turns, stopped, audit=None):
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
        instructions = (EXTRACTION_PROMPT + "\nEach object's content is a USER turn; its id is an opaque identifier. "
            "An optional assistant_reply field is the final assistant answer for that user turn, supplied only as context. "
            "Only answer_note candidates may quote assistant_reply; never turn assistant guesses into user facts. "
            "Copy that exact id into evidence.turn_id. Never shorten or renumber it. "
            "A turn id is not a fact key. Use user.location, user.name, user.answer_length, or "
            "user.programming_language for those attributes. "
            "Statements such as 'I live in Pune' and 'I prefer concise answers' are durable direct facts, "
            "not one-off tasks. Include clear direct facts as user_stated candidates. "
            "A greeting alone has no durable facts.\nUse this JSON schema:\n")
        # Ollama does not expose a preflight tokenizer for this adapter. Use a small
        # UTF-8 byte ceiling plus template margin, never the main model's tokenizer.
        budget = self.settings.memory_context - 1024 - 1024
        omitted = []
        while True:
            schema = deepcopy(MEMORY_SCHEMA)
            evidence_schema = schema["properties"]["candidates"]["items"]["properties"]["evidence"]["items"]
            evidence_schema["properties"]["turn_id"]["enum"] = [turn["id"] for turn in turns]
            system = instructions + json.dumps(schema)
            content = json.dumps(turns, ensure_ascii=False)
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
        for cap in (512, 1024):
            if stopped():
                return [], turns
            if audit:
                audit.log("memory_context", {"model": self.settings.memory_model,
                    "budget_kind": "utf8-byte-ceiling-with-template-margin", "prompt_bytes": size,
                    "byte_ceiling": budget, "context": self.settings.memory_context,
                    "output_cap": cap, "omitted_turn_ids": omitted, "cloud": self.cloud,
                    "assistant_reply_chars": len(turns[-1].get("assistant_reply", "")),
                    "keep_alive": self.settings.memory_keep_alive})
            request = {"model": self.settings.memory_model, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": content}],
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
                audit.log("memory_generation", {"model": self.settings.memory_model,
                    "prompt_tokens": result.get("prompt_eval_count"), "output_tokens": result.get("eval_count"),
                    "finish_reason": result.get("done_reason"), "cloud": self.cloud,
                    "load_duration_ns": result.get("load_duration"),
                    "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
                    "eval_duration_ns": result.get("eval_duration"),
                    "total_duration_ns": result.get("total_duration")})
            try:
                if not result.get("done") or result.get("done_reason") == "length":
                    continue
                data = json.loads("".join(chunks))
                if (isinstance(data, dict) and set(data) == {"candidates"}
                        and isinstance(data["candidates"], list) and len(data["candidates"]) <= 5
                        and all(valid_candidate(candidate) for candidate in data["candidates"])):
                    return data["candidates"], turns
            except (ValueError, TypeError):
                pass
        raise ValueError("Curation returned incomplete or invalid JSON; no new memories were saved")


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
                    candidates, turns = self.extractor.extract(job["turns"], lambda: not self._current(epoch), self.audit)
                    if self._current(epoch):
                        records = self.memory.save_candidates(candidates, turns, job["workspace"],
                                                             sources=job.get("sources", []),
                                                             observed_at=datetime.fromtimestamp(job["created"], timezone.utc).isoformat(),
                                                             before_commit=lambda: self._current(epoch))
                        if self.audit:
                            self.audit.log("memory_write", {"ids": [record["id"] for record in records],
                                           "model": self.settings.memory_model, "background": True})
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
