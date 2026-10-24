from datetime import datetime, timedelta, timezone
import json

from conftest import FakeBackend, embed, tool_call
from harness.curation import MemoryCurator, source_references
from harness.memory import MemoryStore
from harness.memory_view import render_memories
from harness.runtime import AgentRuntime


def proposed_note(quote, key="research.widget.latest"):
    return {"statement": "The assistant reported Widget 2 as the latest release.",
            "fact_key": key, "kind": "answer_note", "confidence": .95,
            "evidence": [{"turn_id": "u1", "quote": quote}]}


def test_answer_note_without_sources_is_dated_unverified_and_expires(settings):
    clock = [datetime(2026, 10, 24, tzinfo=timezone.utc)]
    memory = MemoryStore(settings.state_dir / "memory", embed, now=lambda: clock[0])
    turn = {"id": "u1", "content": "What is the latest Widget release?", "assistant_reply": "Widget 2 is the latest release."}
    try:
        record, = memory.save_candidates([proposed_note(turn["assistant_reply"])], [turn], str(settings.workspace))
        assert record["kind"] == "answer_note" and record["confidence"] == .85
        assert record["verification"] == "assistant_reported" and record["requires_revalidation"]
        assert record["observed_at"] == clock[0].isoformat()
        assert record["related_sources"] == [] and record["evidence"][0]["role"] == "assistant"
        assert record["fact_key"].startswith("answer_note.")
        recalled, = memory.retrieve("latest Widget release", str(settings.workspace))
        assert recalled["requires_revalidation"] and recalled["tentative"]
        assert recalled["user_prompt"] == turn["content"]
        rendered = render_memories([record], color=True, detail=True)
        assert "ANSWER NOTE" in rendered and "verify before reuse" in rendered
        assert "Source URLs unavailable" in rendered and "Assistant quote" in rendered
        clock[0] += timedelta(days=31)
        assert memory.retrieve("latest Widget release", str(settings.workspace)) == []
    finally:
        memory.close()


def test_notes_cannot_quote_tool_text_or_become_user_facts(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    turn = {"id": "u1", "content": "What is the latest release?", "assistant_reply": "Widget 2 is latest."}
    try:
        assert memory.save_candidates([proposed_note("TOOL_ONLY_TEXT")], [turn], "workspace") == []
        fact = proposed_note(turn["assistant_reply"])
        fact["kind"] = "user_stated"
        assert memory.save_candidates([fact], [turn], "workspace") == []
        personal = {"id": "u1", "content": "What language should I learn?", "assistant_reply": "You use Python."}
        guess = proposed_note("You use Python.")
        guess["statement"] = "The user uses Python."
        assert memory.save_candidates([guess], [personal], "workspace") == []
        explicit = memory.save("I prefer concise answers", "explicit", [{"turn_id": "old", "quote": "I prefer concise answers"}],
                               "workspace", key="user.answer_length")
        note = proposed_note(turn["assistant_reply"], key="user.answer_length")
        saved, = memory.save_candidates([note, note], [turn], "workspace")
        assert saved["kind"] == "answer_note"
        assert memory.find(explicit["id"])["status"] == "active"
        assert len(memory.records()) == 2
    finally:
        memory.close()


def test_sources_are_captured_locally_from_successful_tools_only():
    result = {"status": "ok", "content": json.dumps({"results": [
        {"url": "https://example.com/releases", "content": "RAW_RESULT_BODY"},
        {"url": "https://user:password@example.com/private"}]}),
        "artifact_id": "a" * 64, "source_captured_at": "2026-10-24T00:00:00+00:00"}
    trace = [{"type": "tool", "name": "perform_web_search", "result": result},
             {"type": "tool", "name": "fetch_url", "result": {"status": "error", "source": "https://bad.example"}}]
    sources = source_references(trace)
    assert len(sources) == 1 and sources[0]["url"] == "https://example.com/releases"
    assert "RAW_RESULT_BODY" not in str(sources) and "password" not in str(sources)


def test_runtime_queues_sources_locally_and_notes_filter_works(settings):
    class Extractor:
        received = None

        def extract(self, turns, stopped, audit=None):
            self.received = turns
            note = proposed_note(turns[-1]["assistant_reply"])
            note["evidence"][0]["turn_id"] = turns[-1]["id"]
            return [note], turns

        def close(self):
            pass

    class Tools:
        schemas = []
        memory = None

        def execute(self, name, args):
            return {"status": "ok", "content": json.dumps({"results": [{"url": "https://example.com/releases"}]}),
                    "source_captured_at": "2026-10-24T00:00:00+00:00", "artifact_id": "a" * 64}

    memory = MemoryStore(settings.state_dir / "memory", embed)
    extractor = Extractor()
    curator = MemoryCurator(settings, memory, extractor=extractor)
    backend = FakeBackend([tool_call("perform_web_search", {"query": "Widget releases"}), "Widget 2 is the latest release."])
    runtime = AgentRuntime(settings, backend, memory=memory, curator=curator, tools=Tools())
    try:
        runtime.turn("What is the latest Widget release?")
        assert curator.wait_idle(3)
        assert "https://example.com" not in str(extractor.received)
        record, = memory.records()
        assert record["related_sources"][0]["url"] == "https://example.com/releases"
        assert "ANSWER NOTE" in runtime.turn("/memories notes")
        assert "Historical assistant answer" in runtime.turn("/memories " + record["id"])
        assert memory.find(record["id"])["verification"] == "assistant_reported"
    finally:
        runtime.close()
        memory.close()
