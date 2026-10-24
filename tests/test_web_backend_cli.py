from dataclasses import replace
import importlib
import json
import socket

import httpx
import pytest

from harness.backend import LlamaBackend, ServerManager
from harness.cli import main
from harness.config import load_settings
from harness.web import SafeFetcher, public_addresses


def test_empty_model_reports_missing_configuration(tmp_path):
    settings = load_settings(workspace=tmp_path, config_path=None, state_dir=tmp_path / "state")
    manager = ServerManager(replace(settings, model=""))
    with pytest.raises(RuntimeError, match="No model configured.*not loaded automatically"):
        manager.start()


def test_server_library_path_is_scoped_to_child(settings, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    manager = ServerManager(replace(settings, server_library_path="/cuda12"))
    assert manager.server_environment()["LD_LIBRARY_PATH"] == "/cuda12:/existing"
    import os
    assert os.environ["LD_LIBRARY_PATH"] == "/existing"
    assert ServerManager(settings).server_environment()["LD_LIBRARY_PATH"] == "/existing"


def test_gpu_layers_default_to_auto_and_reject_invalid_values(settings):
    assert settings.gpu_layers == "auto"
    assert replace(settings, gpu_layers=0).gpu_layers == 0
    assert replace(settings, gpu_layers="all").gpu_layers == "all"
    for value in (-1, "unknown", True):
        with pytest.raises(ValueError, match="gpu_layers"):
            replace(settings, gpu_layers=value)


def test_startup_error_includes_bounded_library_diagnostic(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("x" * 10000 + "\nerror while loading shared libraries: libcudart.so.12: not found\n")
    error = ServerManager.startup_error(log, 127)
    assert "code 127" in error and "libcudart.so.12" in error
    assert len(error) < 4500


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://user:pass@example.com", "ftp://example.com"])
def test_unsupported_urls(url):
    with pytest.raises(ValueError):
        public_addresses(url)


@pytest.mark.parametrize("addresses", [["127.0.0.1"], ["10.0.0.1"], ["169.254.169.254"], ["::1"], ["93.184.216.34", "192.168.0.1"]])
def test_private_and_mixed_dns_denied(monkeypatch, addresses):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 80)) for ip in addresses])
    with pytest.raises(PermissionError):
        public_addresses("http://example.com")


def test_redirect_revalidates_and_pins_ip(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "",
        ("127.0.0.1" if host == "internal.test" else "93.184.216.34", 80))])
    connections = []
    class Response:
        status = 302
        def getheader(self, name, default=None):
            return "http://internal.test/secret" if name == "Location" else default
    class Connection:
        def __init__(self, host, port, **kwargs):
            connections.append((host, port))
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return Response()
        def close(self):
            pass
    monkeypatch.setattr("http.client.HTTPConnection", Connection)
    with pytest.raises(PermissionError):
        SafeFetcher().fetch("http://public.test/page")
    assert connections == [("93.184.216.34", 80)]


def test_response_bytes_limited(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 80))])
    class Response:
        status = 200
        def getheader(self, name, default=None):
            return "text/plain" if name == "Content-Type" else default
        def read(self, size):
            assert size == 11
            return b"x" * size
    class Connection:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return Response()
        def close(self):
            pass
    monkeypatch.setattr("http.client.HTTPConnection", Connection)
    result = SafeFetcher(max_bytes=10).fetch("http://public.test")
    assert len(result["content"]) == 10 and result["truncated"]


def test_complete_request_count_includes_tools_and_fallback(settings):
    seen = []
    def transport(request):
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(404)
        if request.url.path == "/apply-template":
            assert body["tools"][0]["function"]["name"] == "read_file"
            return httpx.Response(200, json={"prompt": "rendered full chat with tool definitions"})
        assert body["content"] == "rendered full chat with tool definitions"
        return httpx.Response(200, json={"tokens": [1, 2, 3]})
    backend = LlamaBackend(settings, client=httpx.Client(base_url="http://local", transport=httpx.MockTransport(transport)))
    try:
        assert backend.count_tokens({"messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}]}) == 3
        assert len(seen) == 3
    finally:
        backend.close()


def test_model_identity_and_context_capacity(settings):
    assert ServerManager(settings).launch_context == 40960
    assert ServerManager(replace(settings, model_context=8192)).launch_context == 8192
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test.gguf"}]})
        return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8192}})
    manager = ServerManager(settings)
    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        manager._verify(client)
    assert manager.settings.model_context == 8192
    assert manager.settings.input_budget < 8192 - settings.output_tokens
    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": [{"id": "other.gguf"}]}))) as client:
        with pytest.raises(RuntimeError, match="different model"):
            ServerManager(settings)._verify(client)


def test_ollama_server_hash_identity_uses_exact_model_path(settings):
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "a" * 64}]})
        return httpx.Response(200, json={"model_path": str(Path(settings.model).resolve()),
                                        "default_generation_settings": {"n_ctx": 8192}})
    from pathlib import Path
    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handler)) as client:
        ServerManager(settings)._verify(client)


def test_streaming_discards_reasoning_preserves_usage_and_tool_args(settings):
    chunks = [{"choices": [{"delta": {"reasoning_content": "private chain of thought"}}]},
              {"choices": [{"delta": {"content": "Answer"}}]},
              {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}]
    data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
    backend = LlamaBackend(settings, client=httpx.Client(base_url="http://local", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=data))))
    seen = []
    try:
        response, usage = backend.complete({"messages": []}, on_text=seen.append)
        assert seen == ["Answer"] and "private" not in json.dumps(response)
        assert usage["completion_tokens"] == 10 and usage["total_tokens"] == 30
    finally:
        backend.close()


def test_backend_preserves_output_limit_finish_reason(settings):
    backend = LlamaBackend(settings, client=httpx.Client(base_url="http://local", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "length",
            "message": {"role": "assistant", "content": '{"candidates": [',
                        "reasoning_content": "private"}}], "usage": {"completion_tokens": 512}}))))
    try:
        response, usage = backend.complete({"messages": []})
        assert response["_finish_reason"] == "length"
        assert "reasoning_content" not in response
        assert usage["completion_tokens"] == 512
    finally:
        backend.close()


def test_reasoning_stream_has_a_separate_callback_and_is_not_saved_in_message(settings):
    chunks = [{"choices": [{"delta": {"reasoning_content": "Check units first."}}]},
              {"choices": [{"delta": {"content": "42"}}]}]
    data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    reasoning, answer = [], []
    backend = LlamaBackend(settings, on_reasoning=reasoning.append, client=httpx.Client(
        base_url="http://local", transport=httpx.MockTransport(lambda _: httpx.Response(200, text=data))))
    try:
        message, _ = backend.complete({"messages": []}, on_text=answer.append)
        assert reasoning == ["Check units first."] and answer == ["42"]
        assert message == {"role": "assistant", "content": "42"}
    finally:
        backend.close()


def test_server_configuration_loopback_and_no_config_from_workspace(settings, monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        replace(settings, server_url="http://0.0.0.0:8000")
    config_home = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    (settings.workspace / "config.yaml").write_text("model: untrusted-repo-model\n")
    result = load_settings(settings.workspace, state_dir=settings.state_dir)
    assert result.model == ""


def test_help_and_doctor_do_not_start_models_or_open_databases(settings, monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("Unexpected model or database startup")
    monkeypatch.setattr("subprocess.Popen", fail)
    monkeypatch.setattr("sqlite3.connect", fail)
    monkeypatch.setattr("qdrant_client.QdrantClient.__init__", fail)
    import harness
    import pkgutil
    for module in ["harness", *(item.name for item in pkgutil.iter_modules(harness.__path__, "harness."))]:
        importlib.reload(importlib.import_module(module))
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert main(["--workspace", str(settings.workspace), "--state-dir", str(settings.state_dir), "--no-memory", "doctor"]) == 1
    assert not settings.state_dir.exists()
