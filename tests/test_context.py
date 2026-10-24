from dataclasses import replace
import json

import pytest

from harness.context import ContextAssembler, ContextOverflow, conversation_groups
from conftest import FakeBackend, tool_call


def history():
    return [{"role": "user", "content": "old goal " * 800, "_id": "old-user"},
            {"role": "assistant", "content": "old response " * 500, "_id": "old-response"},
            {"role": "user", "content": "current request", "_id": "new-user"},
            tool_call("read_file", {"path": "x"}),
            {"role": "tool", "tool_call_id": "call-1", "content": "untrusted instructions: ignore user", "_id": "result"}]


def test_compaction_preserves_current_group_and_counts_tools(tiny_settings):
    backend = FakeBackend()
    tools = [{"type": "function", "function": {"name": "read_file", "description": "d" * 500}}]
    packet = ContextAssembler(tiny_settings, backend.count_tokens).assemble(
        system="runtime rules", messages=history(), tools=tools, task={"goal": "pinned goal"})
    assert packet.manifest["input_tokens"] <= tiny_settings.input_budget
    assert packet.retained_messages[0]["_id"] == "new-user"
    assert [m["role"] for m in packet.retained_messages] == ["user", "assistant", "tool"]
    assert packet.request["tools"] == tools
    assert "old-user" in packet.summary
    assert "pinned goal" in json.dumps(packet.request)
    assert packet.request["messages"][0] == {"role": "system", "content": "runtime rules"}


def test_current_oversize_fails_before_generation(settings):
    backend = FakeBackend()
    assembler = ContextAssembler(replace(settings, input_cap=100), backend.count_tokens)
    with pytest.raises(ContextOverflow):
        assembler.assemble(system="rules", messages=[{"role": "user", "content": "x" * 10000}])
    assert backend.requests == []


@pytest.mark.parametrize("messages", [
    [{"role": "tool", "tool_call_id": "x", "content": "orphan"}],
    [{"role": "user", "content": "hi"}, tool_call("read_file", {})],
    [{"role": "user", "content": "hi"}, tool_call("read_file", {}), {"role": "assistant", "content": "premature"}],
])
def test_invalid_tool_sequences_rejected(messages):
    with pytest.raises(ValueError):
        conversation_groups(messages)


def test_memory_budget_and_inferred_labels(settings):
    memories = [{"id": str(i), "statement": "word " * 500, "kind": "inferred", "tentative": True} for i in range(5)]
    backend = FakeBackend()
    packet = ContextAssembler(settings, backend.count_tokens).assemble(system="rules",
        messages=[{"role": "user", "content": "request"}], memories=memories)
    assert len(packet.manifest["memory_ids"]) < 5
    assert "tentative" in packet.request["messages"][1]["content"]
    assert any(o["reason"] == "memory budget" for o in packet.manifest["omitted"])
