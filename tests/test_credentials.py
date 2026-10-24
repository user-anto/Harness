from dataclasses import replace
import os

import pytest

from harness.credentials import ollama_api_key
from harness.config import load_settings
from harness.tools import ToolRegistry


def test_env_file_loads_only_ollama_key_without_expansion_or_environment_mutation(settings, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("HARNESS_TEST_UNTRUSTED", raising=False)
    (settings.workspace / ".env").write_text(
        'export OLLAMA_API_KEY="fixture-${UNEXPANDED}"\nHARNESS_TEST_UNTRUSTED=ignored\n')
    assert ollama_api_key(settings) == "fixture-${UNEXPANDED}"
    assert "OLLAMA_API_KEY" not in os.environ
    assert "HARNESS_TEST_UNTRUSTED" not in os.environ
    monkeypatch.setenv("OLLAMA_API_KEY", "exported-fixture")
    assert ollama_api_key(settings) == "exported-fixture"


def test_explicit_env_file_and_missing_key(settings, monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert ollama_api_key(settings) == ""
    path = tmp_path / "credentials.env"
    path.write_text("OLLAMA_API_KEY='explicit-fixture'\n")
    assert ollama_api_key(replace(settings, env_file=str(path))) == "explicit-fixture"
    path.write_text("x" * 65537)
    with pytest.raises(ValueError, match="64 KiB"):
        ollama_api_key(replace(settings, env_file=str(path)))


def test_web_search_uses_env_file_credential(settings, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    (settings.workspace / ".env").write_text("OLLAMA_API_KEY=fixture-search-key\n")
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def web_search(self, **kwargs):
            return {"results": []}

    monkeypatch.setattr("ollama.Client", Client)
    result = ToolRegistry(settings).perform_web_search("test")
    assert result["source"] == "ollama:web_search"
    assert calls[0]["headers"] == {"Authorization": "Bearer fixture-search-key"}


def test_relative_credential_file_resolves_beside_explicit_config(settings, tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "config.yaml"
    config.write_text("env_file: shared.env\n")
    (config_dir / "shared.env").write_text("OLLAMA_API_KEY=config-file-fixture\n")
    selected = load_settings(settings.workspace, config, settings.state_dir)
    assert ollama_api_key(selected) == "config-file-fixture"


def test_generation_reports_current_key_availability_without_exposing_credential(settings, monkeypatch):
    from conftest import FakeBackend
    from harness.runtime import AgentRuntime
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    (settings.workspace / ".env").write_text("OLLAMA_API_KEY=private-fixture-value\n")
    backend = FakeBackend(["Answer"])
    runtime = AgentRuntime(replace(settings, memory_enabled=False), backend)
    try:
        runtime.turn("Search again")
        prompt = backend.requests[0]["messages"][0]["content"]
        assert "web-search credential status: configured" in prompt
        assert "private-fixture-value" not in str(backend.requests)
        assert "Past tool failures describe past attempts" in prompt
    finally:
        runtime.close()
