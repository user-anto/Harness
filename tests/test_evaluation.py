import json
import types

from evals.run_evals import assertions, judge, main


def test_judge_requires_an_exact_label(monkeypatch):
    import ollama
    class Client:
        def __init__(self, **kwargs):
            pass
        def chat(self, **kwargs):
            return types.SimpleNamespace(message=types.SimpleNamespace(content="PASS but actually FAIL"))
    monkeypatch.setattr(ollama, "Client", Client)
    assert judge("fake", "criteria", []) == "INVALID"


def test_report_keeps_failures_and_two_workload_tracks(tmp_path):
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps({"tasks": [
        {"task_id": "coding", "track": "coding-agent", "prompt": "write something", "script": ["done"], "expected_tool_calls": ["write_file"], "expected_files": ["missing.txt"],
         "expected_file_content": {"missing.txt": "27"}},
        {"task_id": "assistant", "track": "general-assistant", "prompt": "hello", "script": ["hello"]}]}))
    output = tmp_path / "reports"
    assert main(["--offline", "--repeats", "1", "--dataset", str(dataset), "--output", str(output)]) == 1
    report = json.loads(next(output.glob("*.json")).read_text())
    assert report["mode"] == "offline-contract"
    assert report["tracks"]["coding-agent"]["passed"] == 0
    assert report["tracks"]["general-assistant"]["passed"] == 1
    assert len(report["results"][0]["failures"]) == 3
    assert report["dataset_sha256"] and report["source_sha256"] and report["packages"]


def test_success_rate_percentiles_use_repeat_rates_not_individual_boolean_runs():
    from evals.run_evals import summarize
    values = []
    for repeat, outcomes in enumerate(((True, False), (True, True), (False, False)), 1):
        for passed in outcomes:
            values.append({"repeat_index": repeat, "passed": passed, "latency_seconds": 2.0,
                           "turn_timings": [{"ttft_seconds": .25}, {"ttft_seconds": None}]})
    metrics = summarize(values)
    assert metrics["completion_rate"] == .5
    assert metrics["completion_rate_p50"] == .5
    assert metrics["completion_rate_p95"] == 1.0
    assert [row["completion_rate"] for row in metrics["repeat_completion_rates"]] == [.5, 1.0, 0.0]
    assert metrics["ttft_sample_count"] == 6 and metrics["p95_ttft_seconds"] == .25
    assert summarize([{**values[0], "turn_timings": []}])["p95_ttft_seconds"] is None


def test_live_ttft_comes_from_stream_callback_not_final_answer(tmp_path, monkeypatch):
    import pytest
    from conftest import FakeBackend
    from evals.run_evals import run_case
    clock = [10.0]
    monkeypatch.setattr("evals.run_evals.time.perf_counter", lambda: clock[0])
    class Manager:
        info = {"fixture": "mock-live-server"}
        def __init__(self, settings):
            self.settings = settings
        def start(self):
            return self.settings
        def close(self):
            pass
    class StreamingBackend(FakeBackend):
        def __init__(self, settings):
            super().__init__(["answer"])
        def complete(self, request, on_text=None):
            assert on_text is not None
            clock[0] += 2
            result = super().complete(request, on_text)
            clock[0] += 3
            return result
        def close(self):
            pass
    monkeypatch.setattr("harness.backend.ServerManager", Manager)
    monkeypatch.setattr("harness.backend.LlamaBackend", StreamingBackend)
    args = types.SimpleNamespace(live=True, config=None, judge=[])
    result = run_case({"task_id": "stream", "track": "general-assistant", "prompt": "Explain streams"}, args)
    assert result["passed"], result["failures"]
    assert result["turn_timings"][0]["ttft_seconds"] == pytest.approx(2)
    assert result["turn_timings"][0]["completion_seconds"] == pytest.approx(5)


def test_security_suite_checks_attempts_even_when_no_side_effect_happens(settings):
    trace = [{"type": "tool", "name": "write_file", "result": {"status": "denied"}}]
    case = {"forbidden_live_tool_calls": ["write_file"], "forbidden_successful_tools": ["write_file"]}
    assert assertions(case, trace, ["denied"], settings.workspace, [], live=False) == []
    assert "followed injected instructions" in assertions(case, trace, ["denied"], settings.workspace, [], live=True)[0]
    assert assertions({"forbidden_answer_substrings": ["CANARY"]}, [], ["canary"], settings.workspace, [])


def test_security_guard_blocks_process_dispatch_if_approval_boundary_regresses(monkeypatch):
    from evals.run_evals import run_case
    monkeypatch.setattr("harness.tools.ToolRegistry._approve", lambda *args: True)
    args = types.SimpleNamespace(live=False, config=None, judge=[])
    case = {"task_id": "guard", "suite": "security", "track": "coding-agent", "prompt": "Try a command",
            "script": [{"call": "run_cmd", "args": {"cmd": "touch NEVER_EXECUTE"}}, "done"],
            "absent_files": ["NEVER_EXECUTE"]}
    result = run_case(case, args)
    assert not result["passed"]
    assert result["process_attempts"] == ["touch NEVER_EXECUTE"]
    assert any("process dispatch" in failure for failure in result["failures"])


def test_all_suites_report_security_categories_metrics_and_repeat_indices(tmp_path, monkeypatch):
    # Exercise suite loading/reporting without rerunning the full benchmark dataset.
    contracts = tmp_path / "contracts.json"
    security = tmp_path / "security.json"
    contracts.write_text(json.dumps({"tasks": [{"task_id": "contract", "track": "general-assistant",
                                              "prompt": "Hello", "script": ["Hello"]}]}))
    categories = ("prompt-injection", "malicious-tool-output", "approval-bypass", "unsafe-command")
    security.write_text(json.dumps({"suite": "security", "tasks": [
        {"task_id": category, "category": category, "track": "coding-agent", "prompt": "Check boundaries",
         "script": ["done"]} for category in categories]}))
    monkeypatch.setattr("evals.run_evals.DEFAULT_DATASET", contracts)
    monkeypatch.setattr("evals.run_evals.SECURITY_DATASET", security)
    output = tmp_path / "reports"
    assert main(["--offline", "--suite", "all", "--repeats", "1", "--output", str(output)]) == 0
    report = json.loads(next(output.glob("*.json")).read_text())
    assert report["schema_version"] == 2
    assert report["suites"]["contracts"]["total"] == 1
    assert report["suites"]["security"]["total"] == 4
    assert set(report["security_categories"]) == {"prompt-injection", "malicious-tool-output", "approval-bypass", "unsafe-command"}
    assert report["metric_definitions"]["ttft_mode"] == "scripted-fixture"
    assert report["tracks"]["coding-agent"]["p95_ttft_seconds"] is not None
    assert all(r["repeat_index"] == 1 for r in report["results"])


def test_schema_v2_baselines_compare_new_metrics_and_reject_old_schema(tmp_path):
    import pytest
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps({"tasks": [
        {"task_id": "greeting", "track": "general-assistant", "prompt": "Hello", "script": ["Hello"]}]}))
    output = tmp_path / "reports"
    argv = ["--offline", "--dataset", str(dataset), "--repeats", "2", "--output", str(output)]
    assert main(argv) == 0
    baseline = next(output.glob("*.json"))
    assert main(argv + ["--baseline", str(baseline)]) == 0
    compared = json.loads(sorted(output.glob("*.json"))[-1].read_text())
    delta = compared["baseline_delta"]["general-assistant"]
    assert delta["completion_rate_p50"] == 0 and delta["completion_rate_p95"] == 0
    assert isinstance(delta["p95_ttft_seconds"], float)
    old = json.loads(baseline.read_text())
    old["schema_version"] = 1
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(old))
    with pytest.raises(SystemExit) as exc:
        main(argv + ["--baseline", str(legacy)])
    assert exc.value.code == 2
