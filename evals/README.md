# Evaluation runner

Active files:

- `benchmarks.json`: coding-agent and general-assistant contract cases.
- `run_evals.py`: offline/live runner and deterministic outcome checks.
- `reports/`: generated private JSON reports, ignored by Git.

## Modes

```bash
make benchmark  # offline scripted contracts, three repetitions
make eval       # explicitly selected local model from config.yaml
.venv/bin/python evals/run_evals.py --offline --repeats 1 --output /tmp/harness-reports
.venv/bin/python evals/run_evals.py --live --config config.yaml --baseline previous.json
```

Offline mode uses scripted replies and fixture embeddings; it does not measure
model intelligence. Live mode uses the configured llama.cpp server and optional
local embedding model. Host shell tools are disabled in live mode; file operations
remain confined to temporary workspaces and use the case's approval policy.
Evaluation runtimes use synchronous extraction for repeatable scripted ordering,
rather than the interactive CLI's background cloud curator.

`--judge MODEL` is live-only and sends redacted traces/answers to the selected
Ollama judge. A cloud judge sends that content off the host. Only exact `PASS` or
`FAIL` responses count; other output fails. Judges are optional and their opinions
do not replace deterministic artifact/tool checks.

## Custom datasets

`--dataset path.json` accepts a JSON object with a nonempty `tasks` list. Every task
must have a unique descriptive `task_id` and an explicit `track` of `coding-agent`
or `general-assistant`; tracks are not guessed from the filename. Supply either
`prompt` or chronological `turns`. Offline mode requires a `script` for every task.

A minimal offline case:

```json
{
  "tasks": [{
    "task_id": "assistant-greeting",
    "track": "general-assistant",
    "prompt": "Hello",
    "script": ["Hello!"],
    "expected_trace_substrings": ["Hello!"]
  }]
}
```

Scripts can use final-answer strings, tool calls such as
`{"call": "evaluate_math", "args": {"expression": "2+2"}}`, and extraction fixtures.
Cases can define `fixtures`, `approve_writes`, `memory`, `history_turns`, and
`restart_after`. Outcome checks include `expected_tool_calls`, `expected_files`,
`expected_file_content`, `absent_files`, `expected_denied_tools`,
`expected_trace_substrings`, `expected_memories`, `memory_count`,
`expected_recalled_memory_count`, and `expect_summary`. See the current dataset for
complete working cases. An answer that claims success cannot satisfy a missing-file
or missing-tool assertion.

Reports contain failures, tool traces, answers, approvals, context manifests,
memories, package/model versions, source/dataset/config hashes, Git revision and
latency. Tracks are reported separately. A baseline must match mode, dataset and
configuration hashes; live comparisons also require matching model identity.
Keep model build, embedding model, hardware and configuration consistent when
interpreting latency differences. Reports can include redacted conversation data;
they are written with private file permissions.

The old v2 golden paths, CSV results and plotting script were removed during
restructuring. Historical scores are not valid current baselines; retrieve the old
files from Git history only when reproducing that version.
