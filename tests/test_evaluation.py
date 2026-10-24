import json
import types

from evals.run_evals import assertions, judge, main


def test_false_success_does_not_pass_artifact_or_tool_assertions(settings):
    case = {"expected_tool_calls": ["write_file"], "expected_files": ["missing"],
            "expected_file_content": {"missing": "27"}}
    failures = assertions(case, [], ["Successfully wrote 27"], settings.workspace, [])
    assert len(failures) == 3


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
        {"task_id": "coding", "track": "coding-agent", "prompt": "write something", "script": ["done"], "expected_files": ["missing.txt"]},
        {"task_id": "assistant", "track": "general-assistant", "prompt": "hello", "script": ["hello"]}]}))
    output = tmp_path / "reports"
    assert main(["--offline", "--repeats", "1", "--dataset", str(dataset), "--output", str(output)]) == 1
    report = json.loads(next(output.glob("*.json")).read_text())
    assert report["mode"] == "offline-contract"
    assert report["tracks"]["coding-agent"]["passed"] == 0
    assert report["tracks"]["general-assistant"]["passed"] == 1
    assert report["results"][0]["failures"]
    assert report["dataset_sha256"] and report["source_sha256"] and report["packages"]
