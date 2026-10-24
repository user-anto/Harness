# Evaluation runner

Harness evaluates two things separately: scripted runtime contracts and the behavior
of an actual configured model. Reports label the mode so fixture timings cannot be
mistaken for model performance.

## Run an evaluation

```bash
make benchmark     # 11 orchestration + 10 security cases, 3 repeats: 63 runs
make security-eval # Only the 10 security cases, 3 repeats: 30 runs
make eval          # Live model, orchestration dataset

# Live security behavior; no host shell commands are executed.
.venv/bin/python evals/run_evals.py --live --suite security --config config.yaml --repeats 3

# Run both suites, or retain a report for later comparisons.
.venv/bin/python evals/run_evals.py --offline --suite all --repeats 10 --output /tmp/harness-reports
.venv/bin/python evals/run_evals.py --offline --suite all --baseline previous.json
```

`--suite contracts` is the runner's default; `make benchmark` explicitly selects
`--suite all`. Each case and repeat gets a fresh temporary workspace and state store.
Normal sessions and memories are not used. Evaluation extraction is synchronous on
the generation backend for repeatability, rather than the interactive background
cloud/local curator. Fallback behavior has separate mocked transport regression tests.

## Metrics and scoring

A run passes only when all its deterministic assertions pass and any optional judge
also returns `PASS`. Exceptions count as failed runs. Metrics are reported separately
by workload track, suite, and security category.

| Metric | Definition |
| --- | --- |
| `passed`, `total`, `completion_rate` | Passing runs, all runs, and their ratio. Rates are fractions from 0 to 1. |
| `repeat_completion_rates` | The pass rate of each complete dataset repeat within the reported group. |
| `completion_rate_p50`, `completion_rate_p95` | Nearest-rank percentiles of those per-repeat pass rates. |
| `p95_ttft_seconds` | Nearest-rank p95 of observed turn-start-to-first-streamed-text delays. |
| `ttft_sample_count`, `ttft_missing_count` | Turns with an observed first text event and turns without one. |
| `p95_seconds` | p95 total case duration, including setup, synchronous extraction, assertions, and an optional judge; cleanup/report writing are excluded. |

For example, repeats scoring 6/10, 8/10, and 10/10 have completion-rate p50 of
0.8 and p95 of 1.0. The overall rate is also 0.8. A high p95 rate describes the
better-performing repeats; it does not describe worst-case reliability. Three
repeats yield coarse estimates, so raise `--repeats` when studying variability.

TTFT starts immediately before `runtime.turn`, including recall, context assembly,
and any tools before the first text. Reasoning, tool-call arguments, headers, and
empty deltas do not count. A local command such as `/remember`, or a failure before
any text, has no streamed TTFT and is represented as `null`, not zero. Those turns
are excluded from TTFT percentiles and counted as missing. Observed TTFTs from
failed runs remain in the sample. Multi-turn cases contribute one sample per turn
that emits text. This is user-visible content latency, not server prefill time.

Offline TTFT uses the scripted callback and a fixture token counter; it measures
orchestration overhead, not GPU/model inference. Live mode enables actual streaming.
Per-turn TTFT and completion duration remain available in each result's `turn_timings`.

## Agent security suite

`security.json` has ten cases across four categories:

| Category | Scenarios |
| --- | --- |
| Prompt injection | Instructions hidden in a repository file or fetched web page. |
| Malicious tool output | Spoofed user/system messages, forged approval, and final-answer hijacking. |
| Approval bypass | Fake `approved` tool arguments, reuse of an earlier approval, and approval claims in files. |
| Unsafe command | Destructive deletion, attempted upload, and command execution through arithmetic. |

**Offline: scripted enforcement.** Scripts deliberately attempt restricted actions;
assertions verify denials, exact resulting artifacts, fresh approval decisions, and
untrusted tool-result labels. A scripted pass does not test model resistance.

**Live: model behavior.** Injection cases require successful reads/fetches to expose
the attack. They check useful answer content and reject injected write/command
attempts even if runtime approvals block them. Direct unsafe requests may be refused
by the model or denied by the approval boundary. The suite is a small set of probes,
not a proof of security or a comprehensive attack-success benchmark.

Security runs exercise the real shell approval path with all shell approvals denied.
A separate process-dispatch guard blocks execution if that boundary regresses and
marks the run failed. File writes use temporary workspaces; only the explicit fresh-
approval case grants its first write. Web pages come from fixtures and external web
search is disabled. No attack command is intentionally executed on the host.
Other live contract evaluations disable shell execution entirely.

## Custom datasets

`--dataset path.json` selects a custom dataset instead of a named suite. Supply a
nonempty `tasks` list, unique `task_id` values, and a `track` of `coding-agent` or
`general-assistant` for each task. Use `prompt` or chronological `turns`. Offline
mode also requires a `script`. A dataset can declare `"suite": "security"` to enable
the process guard and security reporting.

```json
{
  "tasks": [{
    "task_id": "assistant-greeting",
    "track": "general-assistant",
    "prompt": "Hello",
    "script": ["Hello!"],
    "expected_answer_substrings": ["Hello!"]
  }]
}
```

Scripts support final strings, tool calls such as
`{"call": "evaluate_math", "args": {"expression": "2+2"}}`, and extraction fixtures.
Cases can define workspace `fixtures`, `web_fixtures`, `approve_writes`,
`write_approval_sequence`, `memory`, `history_turns`, and `restart_after`.

Assertions include expected tool calls/successes, exact file content, absent files,
expected denials, expected scripted tool-status sequences, answer/trace substrings,
forbidden answer substrings, forbidden successful tools, forbidden live tool attempts,
required untrusted-result labels, exact approval sequences, memories, recall counts,
and context compaction/overflow checks. See both datasets for working examples.
A claim of success cannot satisfy a missing-file or missing-tool assertion.

## Reports, baselines, and optional judges

Private JSON reports go to ignored `evals/reports/`. Schema v2 includes repeat indices,
per-group metrics, security categories, failures, tool traces, answers, approvals,
context manifests, memories, model/package versions, configuration, hashes, and Git
revision. A baseline must match schema, mode, dataset, and configuration hashes;
live comparisons also require matching model identity. Schema v1 reports must be
regenerated before comparison. Baseline deltas cover overall pass rate, completion-
rate percentiles, p95 TTFT, and p95 case duration per workload track.

Keep model build, embedding model, hardware, configuration, and repeat count consistent
when interpreting timings. Reports can contain redacted conversation data and use
private file permissions. Failed cases retain their available timing observations.

`--judge MODEL` is live-only and sends redacted traces/answers to an Ollama judge.
Cloud judges send that content off the host. Only exact `PASS` or `FAIL` responses
count; any other output fails. Judges do not replace artifact/tool assertions.

The old v2 golden paths and CSV tooling were removed. Historical scores are not
current baselines; retrieve old files from Git history only to reproduce that version.
