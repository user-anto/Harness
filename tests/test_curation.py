from dataclasses import replace
import json
import threading

import httpx
import pytest

from conftest import FakeBackend, embed, tool_call
from harness.curation import MemoryCurator, OllamaExtractor, eligible_turn, extraction_turns
from harness.memory import MemoryStore
from harness.runtime import AgentRuntime


def candidate(turn, statement="The user prefers concise answers."):
    return {"statement": statement, "fact_key": "user.answer_length", "kind": "user_stated",
            "confidence": .96, "evidence": [{"turn_id": turn["id"], "quote": turn["content"]}]}


class BlockingExtractor:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def extract(self, turns, stopped, audit=None):
        self.started.set()
        assert self.release.wait(5)
        return [candidate(turns[-1])], turns

    def close(self):
        pass


@pytest.mark.parametrize("text,expected", [("I like Python", True), ("hi", True), ("> quoted", False),
    ("```python\nprint('hello')\n```", False), ("my password is SuperSecret123", False), ("", False)])
def test_turn_eligibility_is_code_enforced(text, expected):
    assert eligible_turn(text) is expected


def test_cpu_ollama_request_uses_schema_and_its_own_context(settings):
    settings = replace(settings, memory_model="llama3.2:3b")
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"llama.context_length": 131072}})
        assert request.url.path == "/api/chat"
        return httpx.Response(200, text=json.dumps({"message": {"content": '{"candidates": []}'},
            "done": True, "done_reason": "stop", "prompt_eval_count": 300, "eval_count": 6}) + "\n")

    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
                                                             transport=httpx.MockTransport(handler)))
    try:
        result, turns = extractor.extract([{"id": "u1", "content": "I live in Pune."}], lambda: False)
        assert result == [] and turns[0]["id"] == "u1"
        payload = requests[-1][1]
        assert payload["model"] == "llama3.2:3b"
        assert payload["options"]["num_gpu"] == 0
        assert payload["options"]["num_thread"] == 2
        assert payload["options"]["num_ctx"] == 8192
        assert payload["keep_alive"] == "10m"
        assert payload["format"]["required"] == ["candidates"]
        evidence = payload["format"]["properties"]["candidates"]["items"]["properties"]["evidence"]
        assert evidence["items"]["properties"]["turn_id"]["enum"] == ["u1"]
        assert "tools" not in payload
    finally:
        extractor.close()


def test_cloud_request_sends_final_reply_without_tools_and_uses_prompt_json(settings):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "thinking"]})
        return httpx.Response(200, text=json.dumps({"message": {"content": '{"candidates": []}'},
            "done": True, "done_reason": "stop"}) + "\n")

    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    turn = {"id": "u1", "content": "I live in Pune.", "assistant_reply": "Understood.",
            "tool_calls": [{"secret_tool": "excluded"}]}
    try:
        assert extractor.extract([turn], lambda: False)[0] == []
        payload = requests[-1]
        assert payload["model"] == "gpt-oss:20b-cloud"
        assert payload["keep_alive"] == "10m" and payload["think"] == "low"
        assert "format" not in payload and "tools" not in payload
        assert "num_gpu" not in payload["options"] and "num_thread" not in payload["options"]
        supplied = json.loads(payload["messages"][1]["content"])
        assert supplied == [{"id": "u1", "content": "I live in Pune.", "assistant_reply": "Understood."}]
        assert "secret_tool" not in str(payload)
    finally:
        extractor.close()


def test_assistant_reply_cannot_supply_memory_evidence(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    try:
        turn = {"id": "u1", "content": "What language should I learn?", "assistant_reply": "You use Python."}
        proposed = candidate(turn, "The user uses Python.")
        proposed["evidence"][0]["quote"] = "You use Python."
        assert memory.save_candidates([proposed], [turn], str(settings.workspace)) == []
        assert memory.records() == []
    finally:
        memory.close()


def test_runtime_queues_only_final_answer_after_a_tool_loop(settings):
    class CaptureCurator:
        received = None

        def submit(self, turns, user_id, workspace, *, assistant_reply="", sources=None):
            self.received = (turns, assistant_reply)

        def close(self):
            pass

    (settings.workspace / "example.txt").write_text("RAW_TOOL_RESULT")
    memory = MemoryStore(settings.state_dir / "memory", embed)
    curator = CaptureCurator()
    backend = FakeBackend([tool_call("read_file", {"path": "example.txt"}), "Here is the final answer."])
    runtime = AgentRuntime(settings, backend, memory=memory, curator=curator)
    try:
        runtime.turn("Read the example file")
        turns, reply = curator.received
        assert reply == "Here is the final answer."
        assert turns[-1]["content"] == "Read the example file"
        assert "RAW_TOOL_RESULT" not in str(curator.received)
        assert "tool_calls" not in str(curator.received)
    finally:
        runtime.close()
        memory.close()


def test_cloud_json_rejects_wrong_field_types_before_saving(settings):
    from harness.curation import valid_candidate
    turn = {"id": "u1", "content": "I live in Pune."}
    assert valid_candidate(candidate(turn))
    for field, value in [("confidence", True), ("confidence", "1.0"), ("statement", 123),
                         ("evidence", []), ("fact_key", "bad key"), ("kind", "explicit")]:
        proposed = candidate(turn)
        proposed[field] = value
        assert not valid_candidate(proposed)


def test_reply_is_redacted_and_budgeted_before_cloud_transmission(settings):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        requests.append(body)
        return httpx.Response(200, text=json.dumps({"message": {"content": '{"candidates": []}'},
            "done": True, "done_reason": "stop"}) + "\n")

    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    try:
        extractor.extract([{"id": "u1", "content": "I live in Pune.",
            "assistant_reply": "password: SuperSecret123\n" + "large answer " * 500}], lambda: False)
        supplied = json.loads(requests[-1]["messages"][1]["content"])
        assert "SuperSecret123" not in str(requests)
        assert supplied[0]["content"] == "I live in Pune."
        assert len(supplied[0]["assistant_reply"]) < 4000
        assert len("".join(m["content"] for m in requests[-1]["messages"]).encode()) <= settings.memory_context - 2048
    finally:
        extractor.close()


def test_background_curation_does_not_delay_the_next_turn(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    extractor = BlockingExtractor()
    curator = MemoryCurator(settings, memory, extractor=extractor)
    backend = FakeBackend(["First answer", "Second answer"])
    runtime = AgentRuntime(settings, backend, memory=memory, curator=curator)
    try:
        assert runtime.turn("I prefer concise answers") == "First answer"
        assert extractor.started.wait(2)
        assert not extractor.release.is_set()
        assert runtime.turn("I prefer brief answers") == "Second answer"
        assert len(backend.requests) == 2
        extractor.release.set()
        assert curator.wait_idle(3)
        assert len(memory.records()) == 1
        assert curator.path.stat().st_mode & 0o777 == 0o600
    finally:
        extractor.release.set()
        runtime.close()
        memory.close()


def test_forget_cancels_inflight_memory_so_it_cannot_reappear(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    record = memory.save("The user prefers concise answers.", "user_stated",
                         [{"turn_id": "old", "quote": "I prefer concise answers"}],
                         str(settings.workspace), key="user.answer_length")
    extractor = BlockingExtractor()
    curator = MemoryCurator(settings, memory, extractor=extractor)
    runtime = AgentRuntime(settings, FakeBackend(["Understood"]), memory=memory, curator=curator)
    try:
        runtime.turn("I prefer concise answers")
        assert extractor.started.wait(2)
        assert runtime.forget_memory(record["id"], "default") == "Memory forgotten."
        extractor.release.set()
        assert curator.wait_idle(3)
        assert memory.records() == []
    finally:
        extractor.release.set()
        runtime.close()
        memory.close()


def test_unfinished_job_survives_restart_without_late_commit(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    first = BlockingExtractor()
    curator = MemoryCurator(settings, memory, extractor=first)
    curator.submit([{"id": "u1", "content": "I prefer concise answers"}], "u1", str(settings.workspace))
    assert first.started.wait(2)
    curator.close()
    assert len(json.loads(curator.path.read_text())) == 1
    first.release.set()
    curator._thread.join(2)
    assert memory.records() == []
    second = BlockingExtractor()
    second.release.set()
    restarted = MemoryCurator(settings, memory, extractor=second)
    try:
        assert restarted.wait_idle(3)
        assert len(memory.records()) == 1
        assert json.loads(restarted.path.read_text()) == []
    finally:
        restarted.close()
        memory.close()


def test_embedding_does_not_hold_the_database_lock(settings):
    started, release = threading.Event(), threading.Event()

    def slow_embed(text):
        if text == "a new fact":
            started.set()
            assert release.wait(3)
        return embed(text)

    memory = MemoryStore(settings.state_dir / "memory", slow_embed)
    memory.save("an existing fact", "explicit", [{"turn_id": "u0", "quote": "an existing fact"}], "workspace")
    worker = threading.Thread(target=lambda: memory.save("a new fact", "user_stated",
        [{"turn_id": "u1", "quote": "a new fact"}], "workspace"))
    worker.start()
    try:
        assert started.wait(2)
        assert len(memory.records()) == 1
        assert not release.is_set()
    finally:
        release.set()
        worker.join(3)
        memory.close()


def test_curation_server_must_be_local(settings):
    with pytest.raises(ValueError, match="memory_server_url"):
        replace(settings, memory_server_url="https://ollama.com")


def test_direct_facts_use_canonical_keys_despite_extractor_variations(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    try:
        for index, city in enumerate(("Pune", "Mumbai")):
            turn = {"id": f"u{index}", "content": f"I live in {city}."}
            fact = candidate(turn, turn["content"])
            fact["fact_key"] = f"u{index}.{city.lower()}"
            assert memory.save_candidates([fact], [turn], str(settings.workspace))
        records = memory.records()
        assert len(records) == 1 and records[0]["fact_key"] == "user.location"
        assert records[0]["statement"] == "I live in Mumbai."
    finally:
        memory.close()


def test_cli_forget_discards_pending_jobs_from_previous_session(settings, monkeypatch):
    from harness.cli import main
    memory = MemoryStore(settings.state_dir / "memory", embed)
    record = memory.save("I live in Pune", "explicit", [{"turn_id": "u1", "quote": "I live in Pune"}],
                         str(settings.workspace))
    pending = settings.state_dir / "curation-pending.json"
    pending.write_text('[{"old": "queued job"}]')
    monkeypatch.setattr("harness.cli.load_settings", lambda *args: settings)
    monkeypatch.setattr("harness.cli.open_memory", lambda settings: memory)
    assert main(["memory", "forget", record["id"]]) == 0
    assert json.loads(pending.read_text()) == []


def test_corrupted_pending_queue_does_not_prevent_chat_startup(settings):
    memory = MemoryStore(settings.state_dir / "memory", embed)
    path = settings.state_dir / "curation-pending.json"
    path.write_text("{invalid json")
    messages = []
    curator = MemoryCurator(settings, memory, extractor=BlockingExtractor(), on_activity=messages.append)
    try:
        assert curator.wait_idle(.1)
        assert any("could not be read" in message for message in messages)
    finally:
        curator.close()
        memory.close()


def test_direct_fact_sends_only_latest_turn_and_inference_gets_one_related_excerpt():
    turns = [{"id": "u0", "content": "I build Python CLI tools for work. " * 30},
             {"id": "u1", "content": "I live in Pune."},
             {"id": "u2", "content": "More Python CLI tools for my next project."}]
    selected = extraction_turns(turns)
    assert [turn["id"] for turn in selected] == ["u0", "u2"]
    assert len(selected[0]["content"]) <= 500
    assert extraction_turns(turns[:2]) == [turns[1]]


def test_greeting_never_loads_the_cpu_model(settings):
    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(lambda request: pytest.fail("Greeting should not call Ollama"))))
    try:
        assert extractor.extract([{"id": "u1", "content": "hi!"}], lambda: False) == ([], [])
    finally:
        extractor.close()


def test_cpu_read_timeout_allows_cold_start(settings):
    extractor = OllamaExtractor(settings)
    try:
        assert extractor.client.timeout.read == 120
        assert extractor.client.timeout.connect == 5
    finally:
        extractor.close()


def test_timeout_retries_once_without_blocking_chat_or_looping_forever(settings):
    class Extractor:
        calls = 0

        def extract(self, turns, stopped, audit=None):
            self.calls += 1
            raise httpx.ReadTimeout("simulated CPU warmup timeout")

        def close(self):
            pass

    memory = MemoryStore(settings.state_dir / "memory", embed)
    extractor = Extractor()
    messages = []
    curator = MemoryCurator(settings, memory, extractor=extractor, on_activity=messages.append)
    try:
        curator.submit([{"id": "u1", "content": "I prefer concise answers"}], "u1", "workspace")
        assert curator.wait_idle(3)
        assert extractor.calls == 2
        assert any("retrying once" in message for message in messages)
        assert any("unavailable" in message for message in messages)
        assert memory.records() == []
    finally:
        curator.close()
        memory.close()
