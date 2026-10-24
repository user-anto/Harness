"""Local model transport and owned process lifecycle, without import side effects."""

import asyncio
import threading
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from urllib.parse import urlsplit

import httpx

from .cancellation import TurnInterrupted
from .config import private_dir
from .security import redact


class LlamaBackend:
    def __init__(self, settings, *, client=None, on_reasoning=None):
        self.settings = settings
        self.client = client
        self._token_counts = {}
        self._injected_client = client is not None
        self._io_loop = None
        self._async_client = None
        self._active_task = None
        self._task_lock = threading.Lock()
        self._interrupt_requested = threading.Event()
        self.on_reasoning = on_reasoning

    def reset_cancel(self):
        self._interrupt_requested.clear()

    def cancel(self):
        self._interrupt_requested.set()
        with self._task_lock:
            if self._active_task is not None:
                self._io_loop.call_soon_threadsafe(self._active_task.cancel)

    def _run(self, coroutine):
        if self._io_loop is None:
            self._io_loop = asyncio.new_event_loop()
        with self._task_lock:
            if self._interrupt_requested.is_set():
                coroutine.close()
                raise TurnInterrupted()
            task = self._io_loop.create_task(coroutine)
            self._active_task = task
        try:
            return self._io_loop.run_until_complete(task)
        except asyncio.CancelledError:
            raise TurnInterrupted() from None
        finally:
            with self._task_lock:
                self._active_task = None

    def _async_transport(self):
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(base_url=self.settings.server_url, timeout=120, trust_env=False)
        return self._async_client

    async def _post_async(self, path, payload):
        return await self._async_transport().post(path, json=payload)

    def _post(self, path, payload):
        if self._injected_client:
            return self.client.post(path, json=payload)
        return self._run(self._post_async(path, payload))

    def close(self):
        if self._async_client is not None:
            self._io_loop.run_until_complete(self._async_client.aclose())
        if self._io_loop is not None:
            self._io_loop.close()
        if self.client is not None:
            self.client.close()

    def count_tokens(self, request):
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if key in self._token_counts:
            return self._token_counts[key]
        count = self._count_tokens(request)
        if len(self._token_counts) >= 256:
            self._token_counts.pop(next(iter(self._token_counts)))
        self._token_counts[key] = count
        return count

    def _count_tokens(self, request):
        response = self._post("/v1/chat/completions/input_tokens", request)
        if response.status_code in {404, 405, 501}:
            # Older llama.cpp builds expose template application and tokenization.
            # They still count the rendered messages AND tools, never characters/4.
            applied = self._post("/apply-template", request)
            applied.raise_for_status()
            prompt = applied.json().get("prompt")
            if not isinstance(prompt, str):
                raise RuntimeError("Server cannot format complete requests for token counting; upgrade llama.cpp")
            counted = self._post("/tokenize", {"content": prompt, "add_special": True})
            counted.raise_for_status()
            return len(counted.json()["tokens"])
        response.raise_for_status()
        count = response.json().get("input_tokens")
        if type(count) is not int or count < 0:
            raise RuntimeError("Invalid token-count response; upgrade llama.cpp")
        return count

    def complete(self, request, on_text=None):
        if self._interrupt_requested.is_set():
            raise TurnInterrupted()
        if on_text is None:
            response = self._post("/v1/chat/completions", {**request, "stream": False})
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
            if self.on_reasoning and reasoning:
                self.on_reasoning(reasoning)
            message = self._clean(choice["message"])
            if choice.get("finish_reason"):
                message["_finish_reason"] = choice["finish_reason"]
            return message, body.get("usage", {})
        payload = {**request, "stream": True, "stream_options": {"include_usage": True}}
        state = {"parts": [], "calls": {}, "usage": {}}
        if self._injected_client:
            with self.client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not self._stream_line(line, state, on_text):
                        break
        else:
            self._run(self._stream_async(payload, state, on_text))
        message = self._clean({"content": "".join(state["parts"]),
                              "tool_calls": [state["calls"][k] for k in sorted(state["calls"])]})
        if state.get("finish_reason"):
            message["_finish_reason"] = state["finish_reason"]
        return message, state["usage"]

    async def _stream_async(self, payload, state, on_text):
        async with self._async_transport().stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not self._stream_line(line, state, on_text):
                    break

    def _stream_line(self, line, state, on_text):
        if self._interrupt_requested.is_set():
            raise TurnInterrupted()
        if not line.startswith("data: "):
            return True
        if line[6:] == "[DONE]":
            return False
        chunk = json.loads(line[6:])
        state["usage"] = chunk.get("usage") or state["usage"]
        choices = chunk.get("choices", [])
        if not choices:
            return True
        if choices[0].get("finish_reason"):
            state["finish_reason"] = choices[0]["finish_reason"]
        delta = choices[0].get("delta", {})
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if self.on_reasoning and reasoning:
            self.on_reasoning(reasoning)
        if delta.get("content"):
            state["parts"].append(delta["content"])
            on_text(delta["content"])
        for call in delta.get("tool_calls", []):
            item = state["calls"].setdefault(call["index"], {"id": "", "type": "function",
                                                          "function": {"name": "", "arguments": ""}})
            if call.get("id"):
                item["id"] = call["id"]
            for key in ("name", "arguments"):
                item["function"][key] += call.get("function", {}).get(key) or ""
        return True

    @staticmethod
    def _clean(message):
        result = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            result["tool_calls"] = message["tool_calls"]
        return result


class ServerManager:
    def __init__(self, settings):
        self.settings, self.process, self.log = settings, None, None
        self.info = {}

    def server_environment(self):
        env = os.environ.copy()
        if self.settings.server_library_path:
            paths = [self.settings.server_library_path, env.get("LD_LIBRARY_PATH", "")]
            env["LD_LIBRARY_PATH"] = os.pathsep.join(path for path in paths if path)
        return env

    @staticmethod
    def startup_error(log_path, returncode):
        with log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 4096))
            tail = stream.read().decode("utf-8", errors="replace").strip()
        return f"llama-server exited (code {returncode}); see {log_path}" + (
            f"\n{redact(tail)}" if tail else "")

    @property
    def launch_context(self):
        # Allocate KV cache only for the configured usable input plus reserves.
        required = math.ceil((self.settings.input_cap + self.settings.output_tokens) / (1 - self.settings.headroom))
        return min(self.settings.model_context, required)

    def _verify(self, client):
        response = client.get("/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        names = [entry["id"] for entry in models]
        expected = Path(self.settings.model).name
        props = client.get("/props")
        props.raise_for_status()
        body = props.json()
        reported_path = body.get("model_path")
        matches = (Path(reported_path).expanduser().resolve() == Path(self.settings.model).expanduser().resolve()
                   if reported_path else any(Path(name).name == expected for name in names))
        if not expected or not matches:
            raise RuntimeError(f"Port is occupied by a different model: expected {expected!r}, got {reported_path or names!r}")
        config = body.get("default_generation_settings", {})
        actual = config.get("n_ctx") or config.get("params", {}).get("n_ctx")
        actual = actual or body.get("n_ctx")
        if not isinstance(actual, int) or actual <= 0:
            raise RuntimeError("Server does not report its context capacity; upgrade llama.cpp")
        trained = [entry.get("meta", {}).get("n_ctx_train") for entry in models]
        trained = [value for value in trained if isinstance(value, int) and value > 0]
        if trained:
            actual = min(actual, min(trained))
        self.settings = replace(self.settings, model_context=min(self.settings.model_context, actual))
        self.info = {"model_ids": names, "model_path": reported_path, "context": actual,
                     "build": body.get("build_info")}

    def start(self):
        if not self.settings.model.strip():
            raise RuntimeError(
                "No model configured. Set model to your GGUF path in a config file and run "
                "harness --config /path/to/config.yaml, or configure ~/.config/harness/config.yaml. "
                "The current directory's config.yaml is not loaded automatically."
            )
        parsed = urlsplit(self.settings.server_url)
        port = parsed.port or 80
        with httpx.Client(base_url=self.settings.server_url, timeout=2, trust_env=False) as client:
            try:
                self._verify(client)
                return self.settings
            except httpx.ConnectError:
                pass
            # Any occupied port, including a non-HTTP process, is an explicit failure.
            with socket.socket(socket.AF_INET6 if parsed.hostname == "::1" else socket.AF_INET) as sock:
                if sock.connect_ex((parsed.hostname, port)) == 0:
                    raise RuntimeError(f"Port {port} is occupied; choose another server_url")
            model = Path(self.settings.model).expanduser().resolve()
            if not model.is_file():
                raise RuntimeError(f"Model file does not exist: {model}; set model in --config")
            server = os.getenv("HARNESS_LLAMA_SERVER_BIN") or shutil.which("llama-server")
            if not server:
                candidate = Path("/usr/local/lib/ollama/llama-server")
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    server = str(candidate)
            if not server:
                raise RuntimeError("Install llama-server or set HARNESS_LLAMA_SERVER_BIN")
            private_dir(self.settings.state_dir)
            log_path = self.settings.state_dir / "llama-server.log"
            self.log = open(log_path, "w", encoding="utf-8")
            log_path.chmod(0o600)
            command = [server, "-m", str(model), "--alias", model.name, "--host", parsed.hostname,
                       "--port", str(port), "--ctx-size", str(self.launch_context),
                       "-ngl", str(self.settings.gpu_layers), "--parallel", "1", "--jinja"]
            try:
                self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                                env=self.server_environment())
                deadline = time.monotonic() + self.settings.server_timeout
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(self.startup_error(log_path, self.process.returncode))
                    try:
                        self._verify(client)
                        return self.settings
                    except (httpx.ConnectError, httpx.HTTPStatusError):
                        time.sleep(.2)
                raise RuntimeError(f"llama-server startup timed out; see {log_path}")
            except BaseException:
                self.close()
                raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()
