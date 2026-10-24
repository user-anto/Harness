"""Missing research notes must recover or explain their rejection."""

from copy import deepcopy
import json
import threading

import httpx
import pytest

from conftest import FakeBackend, embed
from harness.audit import AuditLogger
from harness.curation import MemoryCurator, OllamaExtractor
from harness.memory import MemoryStore
from harness.memory_policy import answer_note_expected, candidate_field_errors, note_excerpts, resolve_note_candidate
from harness.runtime import AgentRuntime


QUESTION = "Could you search the web for an incident where a new exoplant could be discovered with Claude?"
QUOTE = "The reported signal is a possible exoplanet candidate, not a confirmed discovery."
ANSWER = ("The search results describe an amateur astronomer using Claude Code and Codex to analyze TESS data. "
          "The analysis identified repeated brightness dips around a star. " + QUOTE + " "
          "Further observations are required to exclude alternative explanations and establish whether a planet is present. "
          "This is a historical report of the assistant's research answer and needs verification before reuse.")
TURN = {"id": "u1", "content": QUESTION, "assistant_reply": ANSWER}


def note(turn=TURN):
    return {"statement": "The assistant reported an unconfirmed TESS exoplanet candidate found with AI assistance.",
            "fact_key": "research.ai_exoplanet_candidate", "kind": "answer_note", "confidence": .95,
            "evidence": [{"turn_id": turn["id"], "quote": QUOTE}]}


def stream(candidates):
    return httpx.Response(200, text=json.dumps({"done": True, "done_reason": "stop",
        "message": {"content": json.dumps({"candidates": candidates})}}) + "\n")


def focused_selection(body, *, bad_evidence=False):
    turn = json.loads(body["messages"][-1]["content"])[-1]
    source = next(e for e in turn["answer_excerpts"] if e["quote"] == QUOTE)
    return {"statement": note()["statement"], "confidence": .95,
            "evidence_id": "invented" if bad_evidence else source["id"]}


def run_job(settings, handler, turn=None):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    audit = AuditLogger(settings.state_dir / "audit")
    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    messages = []
    curator = MemoryCurator(settings, memory, audit=audit, extractor=extractor, on_activity=messages.append)
    turn = turn or deepcopy(TURN)
    try:
        curator.submit([turn], turn["id"], str(settings.workspace), assistant_reply=turn["assistant_reply"],
                       sources=[{"url": "https://example.invalid/report"}])
        assert curator.wait_idle(5)
        records = memory.records()
        events = [json.loads(line) for line in audit.path.read_text().splitlines()]
        return records, events, messages
    finally:
        curator.close()
        memory.close()


def test_original_invalid_then_empty_case_recovers_via_local_fallback(settings):
    chats = []
    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        chats.append(body)
        if len(chats) == 1:
            invalid = note()
            invalid["statement"] = "PRIVATE_CANDIDATE_TEXT " * 100
            return stream([invalid])
        if body["model"] == settings.memory_model:
            return stream([])
        assert body["options"]["num_gpu"] == 0
        return stream([focused_selection(body)])
    records, events, messages = run_job(settings, handler)
    record, = records
    assert len(chats) == 4  # Initial invalid + empty, focused cloud, focused local.
    assert record["kind"] == "answer_note" and record["requires_revalidation"]
    assert record["verification"] == "assistant_reported" and record["confidence"] == .85
    assert "unconfirmed" in record["statement"]
    assert record["evidence"][0]["quote"] == QUOTE and record["evidence"][0]["role"] == "assistant"
    assert record["related_sources"][0]["url"] == "https://example.invalid/report"
    assert not any("unavailable" in message for message in messages)
    invalid = next(e["content"] for e in events if e["event_type"] == "memory_extraction_invalid")
    assert {"field": "statement", "reason": "invalid_length"} in invalid["candidate_errors"][0]["errors"]
    assert "PRIVATE_CANDIDATE_TEXT" not in json.dumps(events)
    assert any(e["event_type"] == "memory_note_retry" for e in events)
    writes = [e["content"] for e in events if e["event_type"] == "memory_write"]
    assert writes[0]["ids"] == [] and writes[-1]["model"] == settings.memory_fallback_model


@pytest.mark.parametrize("rejection", ["evidence", "confidence"])
def test_policy_rejections_are_logged_and_focused_cloud_retry_can_recover(settings, rejection):
    chats = []
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        chats.append(body)
        proposed = note()
        if len(chats) == 1:
            if rejection == "evidence":
                proposed["evidence"][0]["quote"] = "HALLUCINATED_EVIDENCE"
            else:
                proposed["confidence"] = .6
        return stream([proposed] if len(chats) == 1 else [focused_selection(body)])
    records, events, _ = run_job(settings, handler)
    assert len(records) == 1 and len(chats) == 2
    rejected = next(e["content"] for e in events if e["event_type"] == "memory_candidate_rejected")
    expected = "assistant_quote_or_turn_mismatch" if rejection == "evidence" else "below_threshold"
    assert any(error["reason"] == expected for error in rejected["errors"])
    assert expected in chats[1]["messages"][0]["content"]
    assert chats[1]["options"]["num_predict"] == 1024
    assert "HALLUCINATED_EVIDENCE" not in json.dumps(events)


def test_recovery_does_not_duplicate_or_discard_accepted_user_facts(settings):
    turn = {**TURN, "content": "I prefer concise answers. " + QUESTION}
    chats = []
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        chats.append(body)
        if len(chats) == 1:
            return stream([{"statement": "I prefer concise answers", "fact_key": "user.answer_length",
                "kind": "user_stated", "confidence": .98,
                "evidence": [{"turn_id": "u1", "quote": "I prefer concise answers"}]}])
        return stream([focused_selection(body)])
    records, _, _ = run_job(settings, handler, turn)
    assert len(records) == 2 and len(chats) == 2
    assert {r["kind"] for r in records} == {"user_stated", "answer_note"}


@pytest.mark.parametrize("bad_quote", [False, True])
def test_exhausted_note_recovery_is_bounded_visible_and_never_weakens_evidence(settings, bad_quote):
    chats = []
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        chats.append(body)
        if bad_quote:
            if len(chats) > 1:
                return stream([focused_selection(body, bad_evidence=True)])
            proposed = note()
            proposed["evidence"][0]["quote"] = "TOOL_ONLY_CONTENT"
            return stream([proposed])
        return stream([])
    records, events, messages = run_job(settings, handler)
    assert records == [] and len(chats) == 3
    assert any("Research answer note was not saved" in message for message in messages)
    assert any(e["event_type"] == "memory_error" for e in events)
    assert "TOOL_ONLY_CONTENT" not in json.dumps(events)


@pytest.mark.parametrize("reply", ["Hello!", "I cannot access the web right now. " * 20,
                                  "You use Python and you have several projects. " * 20])
def test_non_substantive_or_personal_guess_answers_do_not_force_recovery(settings, reply):
    assert not answer_note_expected([{**TURN, "assistant_reply": reply}])
    chats = []
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        chats.append(json.loads(request.content))
        return stream([])
    _, events, messages = run_job(settings, handler, {**TURN, "assistant_reply": reply})
    assert len(chats) == 1 and not any(e["event_type"] == "memory_note_retry" for e in events)
    assert not any("unavailable" in message for message in messages)


def test_synchronous_extraction_recovers_missing_note(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    def recovery(request):
        assert "Recover one missing answer_note" in request["messages"][0]["content"]
        return json.dumps({"candidates": [focused_selection(request)]})
    backend = FakeBackend([ANSWER, '{"candidates": []}', recovery])
    runtime = AgentRuntime(settings, backend, memory=memory)
    try:
        assert runtime.turn(QUESTION) == ANSWER
        record, = memory.records()
        assert record["kind"] == "answer_note" and not runtime.warnings
        assert len(backend.requests) == 3
    finally:
        runtime.close()
        memory.close()


def test_cancelled_note_recovery_cannot_fallback_or_save(settings):
    chats, holder = [], {}
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        chats.append(json.loads(request.content))
        if len(chats) == 2:
            holder["curator"].invalidate()
        return stream([])
    memory = MemoryStore(settings.state_dir / "memory", embed)
    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    curator = MemoryCurator(settings, memory, extractor=extractor)
    holder["curator"] = curator
    try:
        curator.submit([TURN], "u1", str(settings.workspace), assistant_reply=ANSWER)
        assert curator.wait_idle(3)
        assert len(chats) == 2 and all(c["model"] == settings.memory_model for c in chats)
        assert memory.records() == []
    finally:
        curator.close()
        memory.close()


def test_rejection_diagnostics_contain_fixed_paths_and_no_unknown_field_values():
    assert candidate_field_errors({"PRIVATE_FIELD_NAME": "PRIVATE_VALUE"})
    errors = json.dumps(candidate_field_errors({"PRIVATE_FIELD_NAME": "PRIVATE_VALUE"}))
    assert "PRIVATE_FIELD_NAME" not in errors and "PRIVATE_VALUE" not in errors
    assert "missing_field" in errors and "unexpected_fields" in errors


def test_focused_recovery_remains_in_background(settings):
    started, release = threading.Event(), threading.Event()
    chats = []
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        chats.append(body)
        if len(chats) == 1:
            return stream([])
        started.set()
        assert release.wait(3)
        return stream([focused_selection(body)])
    memory = MemoryStore(settings.state_dir / "memory", embed)
    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    curator = MemoryCurator(settings, memory, extractor=extractor)
    backend = FakeBackend([ANSWER, "Still available."])
    runtime = AgentRuntime(settings, backend, memory=memory, curator=curator)
    try:
        assert runtime.turn(QUESTION) == ANSWER
        assert started.wait(3) and not release.is_set()
        assert runtime.turn("hi") == "Still available."
        release.set()
        assert curator.wait_idle(3)
        assert len(memory.records()) == 1
    finally:
        release.set()
        runtime.close()
        memory.close()



def test_recovery_binds_exact_source_and_rejects_unknown_or_forged_provenance():
    excerpts = note_excerpts(ANSWER)
    source = next(e for e in excerpts if e["quote"] == QUOTE)
    selection = {"statement": note()["statement"], "confidence": .95, "evidence_id": source["id"]}
    candidate, errors = resolve_note_candidate(selection, TURN, excerpts)
    assert errors == [] and candidate["evidence"] == [{"turn_id": "u1", "quote": QUOTE}]
    assert candidate["kind"] == "answer_note" and candidate["fact_key"].startswith("answer_note.exchange.")
    changed, _ = resolve_note_candidate({**selection, "statement": "Another historical report."}, TURN, excerpts)
    assert changed["fact_key"] == candidate["fact_key"]
    assert resolve_note_candidate({**selection, "evidence_id": "made-up"}, TURN, excerpts)[1]
    assert resolve_note_candidate({**selection, "evidence": [{"turn_id": "wrong", "quote": "fake"}]}, TURN, excerpts)[1]
    assert resolve_note_candidate({**selection, "confidence": .6}, TURN, excerpts)[1] == [
        {"field": "confidence", "reason": "below_threshold"}]



def test_long_greeting_or_typo_clarification_does_not_trigger_note_recovery():
    assert not answer_note_expected([{**TURN, "content": "hello"}])
    assert not answer_note_expected([{**TURN, "assistant_reply": "That appears to be a typo. " * 20}])
    assert note_excerpts("table row without a full sentence\n" * 12)
