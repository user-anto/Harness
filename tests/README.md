# Offline system tests

Run `make test` after `make install`. Test-created workspaces, Qdrant stores and
checkpoints live in pytest temporary directories. An autouse fixture blocks
outbound socket connections. No model download, cloud judge, host credential, or
running Ollama/llama.cpp server is required. Some tool tests execute bounded host
commands in temporary directories to verify timeouts, exit status and output limits.

| Tests | Coverage |
| --- | --- |
| `test_security_tools.py` | Exact approvals, path/symlink escapes, commands, arithmetic and secrets |
| `test_context.py` | Complete-request budgets, whole turns/tool pairs and overflow |
| `test_memory.py` | Local Qdrant persistence, evidence, contradictions, expiry and forgetting |
| `test_curation.py` | Cloud/local request contracts, automatic CPU fallback, cooldown/recovery, cancellation, queue and restart |
| `test_memory_recovery.py` | Invalid/empty research-note recovery, field/policy diagnostics, CPU fallback, cancellation and nonblocking behavior |
| `test_answer_notes.py` | Dated assistant notes, source metadata and separation from user facts |
| `test_memory_edit.py` | Authorized in-place edits, stable IDs, stale writes and evidence |
| `test_memory_view.py` | Readable colors/details, short IDs and rejection of removed commands |
| `test_memory_browser.py` | Real terminal key input, split layout, search, selection and confirmed deletion |
| `test_session_cli.py` | Fresh IDs, explicit resume and read-only listing without model startup |
| `test_session_browser.py` | First-prompt titles/UNKNOWN, workspace restoration, search, switching and deletion |
| `test_runtime.py` | LangGraph/SQLite loops, restart without replay, memory-off, task evidence, locking and retention |
| `test_tui.py`, `test_ui.py` | Composer, slash menu, streaming, reasoning display, selection and scrolling |
| `test_credentials.py` | Allowlisted .env lookup, exported-key priority and no interpolation |
| `test_set_model.py` | GGUF validation, symlinks and atomic default-model replacement |
| `test_web_backend_cli.py` | DNS/redirect policies, pinned fetches, exact token counts, model identity and inert imports |
| `test_optimization_interrupt.py` | Esc recovery, pre-header cancellation, no tool replay, incremental ANSI wrapping, lazy cards, exact-count caching and curator retries |
| `test_evaluation.py` | Artifact/tool assertions, streamed TTFT, repeat success-rate percentiles, security probes/dispatch guard and reports |

The import-side-effect test checks every installed application module, including
new browser modules. The terminal tests send actual prompt-toolkit pipe input
without requiring a physical terminal. They verify behavior and dimensions;
visual smoke checks are still useful on the intended terminal emulator.

These assertions establish software contracts, not general model correctness.
The separate [evaluation runner](../evals/README.md) reports deterministic contracts
and optional live model behavior. [CI](../docs/development.md#github-actions-workflow)
also builds distributions and checks the installed wheel outside the checkout.


## Keeping the suite focused

Tests are retained for distinct failure paths, rather than to reach a test-count target.
Parametrized security inputs (for example IPv4/IPv6 DNS restrictions, credential
formats and terminal key encodings) exercise different contracts and remain separate.

The redundancy review consolidated these overlaps:

| Removed overlap | Coverage retained in |
| --- | --- |
| Assistant replies reused as user-fact evidence | `test_answer_notes.py` tests rejection of tool quotes, assistant quotes and personal guesses. |
| Generic background-input and footer-only checks | `test_tui.py` exercises the real CLI while curation is blocked, including an editable composer and visible background status. |
| Standalone default HTTP-timeout assertion | The CPU request contract in `test_curation.py` checks transport timeouts alongside schema and placement options. |
| Generic timeout retry with a dummy extractor | The local-fallback timeout regression checks the real extractor, one job retry, final warning and no saved records. |
| Separate streaming usage/reasoning fixtures | One backend stream checks separate reasoning, answer content, fragmented tool arguments and usage together. |
| Direct false-success artifact check | The runner report test verifies missing tools/files/content produce three failures and a failing exit status. |

Equivalent ordinary-text eligibility and short-acknowledgement inputs were removed.
The suite/report-format test uses a small synthetic dataset; full benchmark cases
run through `make benchmark` and the CI benchmark step. This keeps scenario execution
out of a test whose purpose is suite selection and report structure.
