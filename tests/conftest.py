from dataclasses import replace
import hashlib
import json
import math
import re
import socket

import pytest

from harness.config import Settings


@pytest.fixture(autouse=True)
def isolated_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user-config"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Offline test attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@pytest.fixture
def settings(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Settings(workspace=workspace, state_dir=tmp_path / "state", model="test.gguf")


def embed(text):
    vector = [0.0] * 128
    for word in re.findall(r"\w+", text.lower()):
        index = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % len(vector)
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1
    return [v / norm for v in vector]


class FakeBackend:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.requests = []

    def count_tokens(self, request):
        # Stable fixture tokenizer, not a claim about real model token counts.
        return len(json.dumps(request).encode()) // 4 + 1

    def complete(self, request, on_text=None):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Unexpected model call")
        result = self.responses.pop(0)
        if callable(result):
            result = result(request)
        if isinstance(result, str):
            result = {"role": "assistant", "content": result}
        if on_text and result.get("content"):
            on_text(result["content"])
        return result, {"prompt_tokens": self.count_tokens(request), "completion_tokens": 1}


def tool_call(name, args, ident="call-1"):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": ident, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}


@pytest.fixture
def tiny_settings(settings):
    return replace(settings, model_context=8192, input_cap=2500, output_tokens=512)
