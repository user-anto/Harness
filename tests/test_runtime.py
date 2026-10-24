from dataclasses import replace
import json

import pytest

from harness.memory import MemoryStore
from harness.runtime import AgentRuntime
from conftest import FakeBackend, embed, tool_call


def test_stream_and_final_answer_arrive_before_bounded_memory_extraction(settings):
    events = []

    def extract(request):
        assert ("answer", "Visible answer.") in events
        assert request["max_tokens"] == 512
        assert request["reasoning_effort"] == "none"
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
        assert request["response_format"]["schema"]["required"] == ["candidates"]
        events.append(("extract", "started"))
        return {"role": "assistant", "content": '{"candidates": []}'}

    memory = MemoryStore(settings.state_dir / "memory", embed)
    runtime = AgentRuntime(settings, FakeBackend(["Visible answer.", extract]), memory=memory,
                           on_text=lambda text: events.append(("text", text)),
                           on_answer=lambda text: events.append(("answer", text)),
                           on_status=lambda status: events.append(("status", status)))
    try:
        assert runtime.turn("Hello") == "Visible answer."
        assert "reasoning_effort" not in runtime.backend.requests[0]
        assert "chat_template_kwargs" not in runtime.backend.requests[0]
        assert events.index(("text", "Visible answer.")) < events.index(("extract", "started"))
        assert ("status", "memory") in events
    finally:
        runtime.close()
        memory.close()


def test_truncated_extraction_retries_once_and_never_saves_partial_candidates(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    backend = FakeBackend(["Hello!", {"role": "assistant", "content": '{"candidates": [',
                                    "_finish_reason": "length"}, '{"candidates": []}'])
    runtime = AgentRuntime(settings, backend, memory=memory)
    try:
        assert runtime.turn("hi") == "Hello!"
        assert not runtime.warnings and memory.records() == []
        assert [request["max_tokens"] for request in backend.requests[1:]] == [512, 1024]
    finally:
        runtime.close()
        memory.close()


@pytest.mark.parametrize("bad_json", ['{"candidates": [', '[]', '{}', '{"candidates": {}}'])
def test_invalid_extraction_is_bounded_and_does_not_fail_the_answer(settings, bad_json):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    backend = FakeBackend(["Hello!", bad_json, bad_json])
    runtime = AgentRuntime(settings, backend, memory=memory)
    try:
        assert runtime.turn("hi") == "Hello!"
        assert len(backend.requests) == 3 and memory.records() == []
        assert len(runtime.warnings) == 1
        assert "no new memories were saved" in runtime.warnings[0]
        assert "line " not in runtime.warnings[0]
    finally:
        runtime.close()
        memory.close()


def test_extraction_respects_small_output_setting_without_repeating_identical_request(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    backend = FakeBackend(["Hello!", '{"candidates": ['])
    runtime = AgentRuntime(replace(settings, output_tokens=128), backend, memory=memory)
    try:
        assert runtime.turn("hi") == "Hello!"
        assert len(backend.requests) == 2 and backend.requests[-1]["max_tokens"] == 128
        assert memory.records() == []
    finally:
        runtime.close()
        memory.close()


def test_tool_cycle_denial_and_provenance(settings):
    backend = FakeBackend([tool_call("write_file", {"path": "x", "content": "hello"}), "Write was denied."])
    runtime = AgentRuntime(settings, backend, approve=lambda *a: False)
    try:
        assert runtime.turn("Write hello to x") == "Write was denied."
        assert not (settings.workspace / "x").exists()
        result = json.loads(backend.requests[-1]["messages"][-1]["content"])
        assert result["status"] == "denied" and result["trust"] == "untrusted-data"
        assert runtime.snapshot("default")["manifest"]["input_tokens"] <= settings.input_budget
    finally:
        runtime.close()


def test_resume_does_not_replay_pending_command(settings):
    runtime = AgentRuntime(settings, FakeBackend())
    runtime.graph.update_state(runtime.config("resume"), {"messages": [
        {"role": "user", "content": "do something", "_id": "user"}, tool_call("run_cmd", {"cmd": "touch should-not-exist"})],
        "workspace": str(settings.workspace)})
    runtime.close()
    backend = FakeBackend(["Previous execution outcome is unknown."])
    runtime = AgentRuntime(settings, backend, approve=lambda *a: pytest.fail("Replayed command requested approval"))
    try:
        runtime.turn("What happened?", "resume")
        assert not (settings.workspace / "should-not-exist").exists()
        assert "outcome" in json.dumps(backend.requests[0])
        tool_results = [m for m in backend.requests[0]["messages"] if m["role"] == "tool"]
        assert json.loads(tool_results[0]["content"])["status"] == "interrupted"
    finally:
        runtime.close()


def test_explicit_memory_acknowledged_only_after_write_and_recalled_each_turn(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    backend = FakeBackend(["You prefer concise answers.", '{"candidates": []}', "Still concise.", '{"candidates": []}'])
    runtime = AgentRuntime(settings, backend, memory=memory)
    try:
        reply = runtime.turn("Please remember that I prefer concise answers")
        assert reply.startswith("Remembered:") and len(memory.records()) == 1
        assert backend.requests == []
        runtime.turn("What answers do I prefer?")
        runtime.turn("Keep the answers concise")
        for index in [0, 2]:
            assert "I prefer concise answers" in json.dumps(backend.requests[index])
        extraction_request = backend.requests[1]
        supplied = json.loads(extraction_request["messages"][-1]["content"])
        assert supplied[-1]["assistant_reply"] == "You prefer concise answers."
    finally:
        runtime.close()
        memory.close()


def test_disabled_memory_never_reads_or_writes(settings):
    class Bomb:
        def __getattr__(self, name):
            pytest.fail("Disabled memory was accessed: " + name)
    runtime = AgentRuntime(replace(settings, memory_enabled=False),
        FakeBackend([tool_call("search_memory", {"query": "anything"}), "ok"]), memory=Bomb())
    try:
        assert "disabled" in runtime.turn("/remember I like tea")
        assert runtime.turn("hello") == "ok"
    finally:
        runtime.close()


def test_malformed_tool_args_are_errors_not_empty_execution(settings):
    call = tool_call("run_cmd", {})
    call["tool_calls"][0]["function"]["arguments"] = "{invalid"
    runtime = AgentRuntime(settings, FakeBackend([call, "Could not execute invalid arguments."]),
                           approve=lambda *a: pytest.fail("Malformed arguments reached execution"))
    try:
        runtime.turn("do it")
        assert runtime.last_trace[1]["result"]["status"] == "error"
    finally:
        runtime.close()


def test_task_completion_requires_matching_next_task_and_real_evidence(settings):
    backend = FakeBackend([tool_call("task_complete", {"task_description": "test", "evidence": "made up"}), "Not verified."])
    runtime = AgentRuntime(settings, backend)
    runtime.graph.update_state(runtime.config("default"), {"task": {"goal": "test", "pending": ["test"], "completed": []}})
    try:
        runtime.turn("finish")
        assert runtime.snapshot("default")["task"]["pending"] == ["test"]
        assert runtime.last_trace[1]["result"]["status"] == "error"
    finally:
        runtime.close()


def test_secrets_do_not_enter_checkpoint_or_audit(settings):
    runtime = AgentRuntime(settings, FakeBackend(["ok"]))
    try:
        runtime.turn("My password is SuperPrivate123")
    finally:
        runtime.close()
    for path in settings.state_dir.rglob("*"):
        if path.is_file():
            assert b"SuperPrivate123" not in path.read_bytes()
            assert path.stat().st_mode & 0o077 == 0


def test_workspace_session_mismatch_and_exclusive_runtime(settings, tmp_path):
    runtime = AgentRuntime(settings, FakeBackend(["hello"]))
    runtime.turn("hello")
    with pytest.raises(RuntimeError, match="Another"):
        AgentRuntime(settings, FakeBackend())
    runtime.close()
    other = tmp_path / "other"
    other.mkdir()
    runtime = AgentRuntime(replace(settings, workspace=other), FakeBackend())
    try:
        with pytest.raises(ValueError, match="different workspace"):
            runtime.turn("hello")
    finally:
        runtime.close()


def test_retention_removes_whole_old_sessions(settings):
    runtime = AgentRuntime(settings, FakeBackend(["hello"]))
    runtime.turn("hello", "old")
    runtime.conn.execute("UPDATE session_activity SET touched = ? WHERE id = ?", ("2000-01-01", "old"))
    runtime.conn.commit()
    runtime.close()
    runtime = AgentRuntime(settings, FakeBackend())
    try:
        assert runtime.snapshot("old") == {}
    finally:
        runtime.close()


def test_task_constraints_require_user_quotes_and_persist(settings):
    backend = FakeBackend([tool_call("record_task", {"kind": "constraint", "user_quote": "Use Python 3.12"}), "Recorded."])
    runtime = AgentRuntime(settings, backend)
    try:
        runtime.turn("Use Python 3.12 for this task")
        assert runtime.snapshot("default")["task"]["constraints"] == ["Use Python 3.12"]
    finally:
        runtime.close()
    backend = FakeBackend([tool_call("record_task", {"kind": "decision", "user_quote": "Invented requirement"}), "Not recorded."])
    runtime = AgentRuntime(settings, backend)
    try:
        runtime.turn("Continue")
        assert runtime.snapshot("default")["task"]["decisions"] == []
        assert runtime.last_trace[1]["result"]["status"] == "error"
    finally:
        runtime.close()
