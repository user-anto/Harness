from datetime import datetime, timedelta, timezone

from conftest import FakeBackend, embed
from harness.memory import MemoryStore
from harness.memory_view import render_memories
from harness.runtime import AgentRuntime


def test_memory_cards_use_types_colors_and_explain_tentative_records():
    now = datetime.now(timezone.utc)
    records = [{"id": "abcdef01-1234-4321-abcd-0123456789ab", "statement": "Prefers concise answers",
                "kind": "inferred", "confidence": .8, "updated_at": now.isoformat(),
                "workspace": "/projects/Harness", "fact_key": "user.answer_length",
                "evidence": [{"turn_id": "u1", "quote": "short answers please"},
                             {"turn_id": "u2", "quote": "keep it concise"}],
                "expires_at": (now + timedelta(days=90)).isoformat()}]
    plain = render_memories(records)
    assert "TENTATIVE INSIGHT" in plain and "abcdef01" in plain
    assert "Extractor score 0.80" in plain and "2 supporting turns" in plain
    assert "expires in 90 days" in plain and "\033[" not in plain
    colorful = render_memories(records, color=True, detail=True)
    assert "\033[33m" in colorful and "short answers please" in colorful
    assert "Full ID:" in colorful and "Source: /projects/Harness" in colorful


def test_memory_view_handles_empty_state_and_terminal_escape_in_content():
    assert "/remember" in render_memories([])
    record = {"id": "abcdef01-1234-4321-abcd-0123456789ab", "statement": "A fact\033[31m",
              "kind": "explicit", "confidence": 1, "evidence": [], "workspace": "test"}
    assert "\033[" not in render_memories([record])
    assert "User-confirmed" in render_memories([record])


def test_memories_are_readable_short_ids_open_evidence_and_forget_all_versions(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    runtime = AgentRuntime(settings, FakeBackend(), memory=memory)
    try:
        runtime.turn("/remember I live in Pune")
        record = memory.records()[0]
        ident = record["id"][:8]
        view = runtime.turn("/memories")
        assert "SAVED BY YOU" in view and "I live in Pune" in view
        assert '"statement":' not in view
        detail = runtime.turn("/memories " + ident)
        assert "Evidence:" in detail and record["id"] in detail
        assert memory.find(ident) == record
        assert memory.find(record["id"]) == record
        assert runtime.forget_memory(ident, "default") == "Memory forgotten."
        assert memory.records() == []
        assert runtime.snapshot("default")["messages"] == []
        assert "No saved memories yet" in runtime.turn("/memories")
    finally:
        runtime.close()
        memory.close()


def test_removed_slash_commands_are_not_advertised_or_sent_to_model(settings):
    from harness.commands import COMMANDS, command_help
    assert not {'/inspect', '/forget'} & {item.name for item in COMMANDS}
    assert '/inspect' not in command_help() and '/forget' not in command_help()
    backend = FakeBackend()
    runtime = AgentRuntime(settings, backend)
    try:
        for prompt in ['/inspect abcdef01', '/forget abcdef01']:
            assert 'Unknown command' in runtime.turn(prompt)
        assert backend.requests == []
    finally:
        runtime.close()
