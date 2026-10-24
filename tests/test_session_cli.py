from dataclasses import replace
from io import StringIO

from conftest import FakeBackend
from harness.cli import chat, parser
from harness.ui import TerminalUI


def test_cli_defaults_to_new_session_and_named_sessions_resume_explicitly(settings, monkeypatch):
    settings = replace(settings, memory_enabled=False)
    backends = []

    class Backend(FakeBackend):
        def close(self):
            pass

    def backend_factory(_):
        backend = Backend(["Answer"])
        backends.append(backend)
        return backend

    monkeypatch.setattr("harness.backend.ServerManager.start", lambda self: settings)
    monkeypatch.setattr("harness.backend.LlamaBackend", backend_factory)
    first = parser().parse_args(["--prompt", "First question"])
    second = parser().parse_args(["--prompt", "Second question"])
    stream = StringIO()
    for args in (first, second):
        assert chat(args, settings, TerminalUI(stream=stream, interactive=False)) == 0
    assert first.session != second.session
    second_users = [message["content"] for message in backends[1].requests[0]["messages"]
                    if message["role"] == "user" and not message["content"].startswith("<context_data")]
    assert second_users == ["Second question"]
    resumed = parser().parse_args(["--session", first.session, "--prompt", "Follow-up question"])
    assert chat(resumed, settings, TerminalUI(stream=stream, interactive=False)) == 0
    assert "Resumed saved session: 2 retained messages" in stream.getvalue()
    users = [message["content"] for message in backends[2].requests[0]["messages"]
             if message["role"] == "user" and not message["content"].startswith("<context_data")]
    assert users == ["First question", "Follow-up question"]


def test_session_listing_is_read_only_and_does_not_start_models(settings, monkeypatch, capsys):
    from harness.cli import main
    from harness.runtime import AgentRuntime
    from harness.sessions import list_sessions, render_sessions

    assert list_sessions(settings.state_dir) == []
    assert not settings.state_dir.exists()
    runtime = AgentRuntime(replace(settings, memory_enabled=False), FakeBackend(["Answer"]))
    try:
        runtime.turn("A question", session="my-project")
        listing = runtime.turn("/sessions", session="my-project")
        assert "Current session: my-project" in listing
        assert "--session my-project" in listing
    finally:
        runtime.close()

    records = list_sessions(settings.state_dir)
    assert records[0]["id"] == "my-project"
    assert records[0]["workspace"] == str(settings.workspace)
    assert "--workspace " in render_sessions(records)

    def fail(*args, **kwargs):
        raise AssertionError("Session discovery must not start a model or memory store")

    monkeypatch.setattr("harness.backend.ServerManager.start", fail)
    monkeypatch.setattr("harness.cli.open_memory", fail)
    assert main(["--workspace", str(settings.workspace), "--state-dir", str(settings.state_dir), "sessions"]) == 0
    assert "--session my-project" in capsys.readouterr().out
