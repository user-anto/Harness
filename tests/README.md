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
| `test_curation.py` | Cloud/local request contracts, background queue, retry, invalidation and restart |
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
| `test_evaluation.py` | Artifact/tool assertions, exact judge labels, tracks and report metadata |

The import-side-effect test checks every installed application module, including
new browser modules. The terminal tests send actual prompt-toolkit pipe input
without requiring a physical terminal. They verify behavior and dimensions;
visual smoke checks are still useful on the intended terminal emulator.

These assertions establish software contracts, not general model correctness.
The separate [evaluation runner](../evals/README.md) reports deterministic contracts
and optional live model behavior. [CI](../docs/development.md#github-actions-workflow)
also builds distributions and checks the installed wheel outside the checkout.
