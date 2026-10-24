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


@pytest.mark.parametrize("text,expected", [("I like Python", True), ("> quoted", False),
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

    extractor = OllamaExtractor(settings)
    try:
        assert extractor.client.timeout.read == settings.memory_timeout
        assert extractor.client.timeout.connect == 5
        extractor.client.close()
        extractor.client = httpx.Client(base_url=settings.memory_server_url, transport=httpx.MockTransport(handler))
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
    assert curator.path.stat().st_mode & 0o777 == 0o600
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


@pytest.mark.parametrize("failure", ["connect", "timeout", "status", "missing", "stream", "invalid"])
def test_cloud_failures_use_local_cpu_fallback_with_same_evidence(settings, failure):
    requests, events = [], []
    class Audit:
        def log(self, kind, payload):
            events.append((kind, payload))

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        cloud = body["model"] == settings.memory_model
        if request.url.path == "/api/show":
            return httpx.Response(404 if cloud and failure == "missing" else 200, json={})
        if cloud:
            if failure == "connect":
                raise httpx.ConnectError("offline", request=request)
            if failure == "timeout":
                raise httpx.ReadTimeout("slow cloud", request=request)
            if failure == "status":
                return httpx.Response(503, json={"error": "unavailable"})
            if failure == "stream":
                return httpx.Response(200, text=json.dumps({"error": "cloud unavailable"}) + "\n")
            return httpx.Response(200, text=json.dumps({"done": True,
                "message": {"content": '{"candidates": ['}}) + "\n")
        supplied = json.loads(body["messages"][1]["content"])
        record = candidate(supplied[-1])
        return httpx.Response(200, text=json.dumps({"done": True,
            "message": {"content": json.dumps({"candidates": [record]})}}) + "\n")

    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    turn = {"id": "u1", "content": "I prefer concise answers", "assistant_reply": "Understood."}
    try:
        records, evidence = extractor.extract([turn], lambda: False, Audit())
        assert records == [candidate(turn)] and evidence == [turn]
        assert extractor.last_model == "llama3.2:3b"
        payload = requests[-1][1]
        assert payload["options"]["num_gpu"] == 0 and payload["options"]["num_thread"] == 2
        assert payload["keep_alive"] == "10m" and "format" in payload and "think" not in payload
        assert events[-2][0] == "memory_context" and any(e[0] == "memory_fallback" for e in events)
    finally:
        extractor.close()


def test_fallback_cooldown_and_cloud_recovery(settings, monkeypatch):
    clock, cloud_calls = [100.0], []
    monkeypatch.setattr("harness.curation.time.monotonic", lambda: clock[0])

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        if body["model"] == settings.memory_model:
            cloud_calls.append(clock[0])
            if len(cloud_calls) == 1:
                return httpx.Response(503)
        return httpx.Response(200, text=json.dumps({"done": True,
            "message": {"content": '{"candidates": []}'}}) + "\n")

    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    try:
        turns = [{"id": "u1", "content": "I live in Pune."}]
        extractor.extract(turns, lambda: False)
        assert extractor.last_model == settings.memory_fallback_model
        clock[0] += 1
        extractor.extract(turns, lambda: False)
        assert cloud_calls == [100.0]
        clock[0] += settings.memory_cloud_retry_interval
        extractor.extract(turns, lambda: False)
        assert cloud_calls == [100.0, 161.0]
        assert extractor.last_model == settings.memory_model
    finally:
        extractor.close()


def test_cancelled_cloud_failure_never_starts_fallback(settings):
    stopped, requests = threading.Event(), []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body["model"])
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        stopped.set()
        raise httpx.ConnectError("offline", request=request)
    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    try:
        assert extractor.extract([{"id": "u1", "content": "I live in Pune."}], stopped.is_set) == ([], [])
        assert settings.memory_fallback_model not in requests
    finally:
        extractor.close()


def test_cloud_and_missing_local_fallback_report_failure_without_saving(settings):
    def handler(request):
        return httpx.Response(404, json={"error": "model missing"})
    memory = MemoryStore(settings.state_dir / "memory", embed)
    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    messages = []
    curator = MemoryCurator(settings, memory, extractor=extractor, on_activity=messages.append)
    try:
        curator.submit([{"id": "u1", "content": "I live in Pune."}], "u1", str(settings.workspace))
        assert curator.wait_idle(3)
        assert memory.records() == []
        assert any("local fallback llama3.2:3b failed" in message for message in messages)
        assert any("ollama pull llama3.2:3b" in message for message in messages)
    finally:
        curator.close()
        memory.close()


def test_fallback_cannot_silently_resolve_to_remote_model(settings):
    chats = []
    def handler(request):
        body = json.loads(request.content)
        if body["model"] == settings.memory_model:
            return httpx.Response(503)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"remote_host": "https://example.invalid"})
        chats.append(body)
        return httpx.Response(200)
    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(ValueError, match="resolves to a cloud model"):
            extractor.extract([{"id": "u1", "content": "I live in Pune."}], lambda: False)
        assert not chats
    finally:
        extractor.close()


def test_fallback_can_be_disabled_and_cloud_fallback_names_are_rejected(settings):
    with pytest.raises(ValueError, match="local model"):
        replace(settings, memory_fallback_model="another:cloud")
    extractor = OllamaExtractor(replace(settings, memory_fallback_model=""),
        client=httpx.Client(base_url=settings.memory_server_url,
            transport=httpx.MockTransport(lambda request: httpx.Response(503))))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            extractor.extract([{"id": "u1", "content": "I live in Pune."}], lambda: False)
        assert extractor._fallback is None
    finally:
        extractor.close()


def test_local_fallback_timeout_preserves_one_background_job_retry(settings):
    chats = []
    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        chats.append(body["model"])
        if body["model"] == settings.memory_model:
            return httpx.Response(503)
        raise httpx.ReadTimeout("local cold start", request=request)
    memory = MemoryStore(settings.state_dir / "memory", embed)
    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    messages = []
    curator = MemoryCurator(settings, memory, extractor=extractor, on_activity=messages.append)
    try:
        curator.submit([{"id": "u1", "content": "I live in Pune."}], "u1", str(settings.workspace))
        assert curator.wait_idle(3)
        assert chats == [settings.memory_model, settings.memory_fallback_model, settings.memory_fallback_model]
        assert any("retrying once" in message for message in messages)
        assert any("unavailable" in message for message in messages)
        assert memory.records() == []
    finally:
        curator.close()
        memory.close()
