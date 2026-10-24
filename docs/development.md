# Development, repository structure and CI

## Source map

`harness/` is the only installed application package. There is no parallel `src/`
implementation. Console commands are declared in `pyproject.toml`:

- `harness` → `harness.cli:main`
- `harness-set-model` → `harness.set_model:main`
- `python -m harness` delegates to the same CLI.

| Area | Modules | Responsibility |
| --- | --- | --- |
| Composition | `cli`, `commands`, `config`, `set_model` | Arguments, configuration, command registry, dependency construction |
| Presentation | `tui`, `ui`, `browser`, `memory_browser`, `session_browser`, `memory_view` | Chat/approval output, shared card navigation and browser views |
| Sessions | `runtime`, `sessions` | LangGraph/checkpoints, loading/deletion, metadata and first-prompt titles |
| Context | `context`, `prompts` | Whole-turn budgeting, task record, memory trust and extraction instructions |
| Memory | `memory`, `memory_policy`, `curation` | Qdrant records, shared evidence validation, rejection diagnostics, answer-note recovery and background queue |
| Models | `backend` | llama.cpp requests, full-template token counts and owned-server lifecycle |
| Tools | `tools`, `web` | Workspace/host/memory tools, approvals and bounded public HTTP fetches |
| Boundaries | `security`, `credentials`, `audit` | Path/secret policies, allowlisted credential lookup and redacted metadata |

`tests/` contains offline system tests. `evals/` contains the current benchmark
runner and dataset; generated reports go to ignored `evals/reports/`. `docs/`
contains architecture, prompt structure, memory policy, terminal setup and smoke
instructions. `MANIFEST.in` adds guides, tests, dataset, logo and development files
and the CI workflow to source distributions; `banner.ansi` is package data in the wheel.

## Local verification

```bash
make install
make lint
make test
make benchmark
make security-eval
make build
```

`make install` creates `.venv` and installs an editable development package using
`requirements.lock` as dependency constraints. `pyproject.toml` declares supported
ranges; the constraints record tested versions. Run `source .venv/bin/activate`
before using plain `harness` and `harness-set-model` commands, or invoke their
`.venv/bin/` paths directly. Installation does not modify shell startup files or
create global wrappers. Tests use temporary workspaces and
state, fake model transports, and a fixture that blocks outbound socket connections.
Real local Qdrant and SQLite are exercised. See [test coverage](../tests/README.md)
and [evaluation modes](../evals/README.md).

`make eval` is a separate, opt-in local-model run. It is not a CI prerequisite.
Live contract runs disable shell execution. Security runs exercise denied shell
approvals behind a separate process-dispatch guard, use fixture web pages, and report
injected tool attempts as live behavior failures. `make benchmark` includes both
suites; `make security-eval` selects only security cases. Neither an offline pass nor an optional
model judge proves general intelligence or prompt-injection resistance.

## GitHub Actions workflow

The actual file is [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
GitHub accepts both `.yml` and `.yaml`; this repository uses `.yml`.
CI means continuous integration: every push and pull request runs the same
verification on an Ubuntu runner, separately for Python 3.12 and 3.13.

| Step | What it does | Why it exists |
| --- | --- | --- |
| Checkout | Fetch the proposed repository contents | Test the actual revision |
| Set up Python | Select each matrix Python version | Catch interpreter compatibility failures |
| Install | Install `.[dev]` with `requirements.lock` constraints | Reproduce tested dependency versions |
| Lint | Run Ruff on application, tests and runner | Catch undefined names, unused imports and syntax issues |
| Offline tests | Run the complete pytest suite | Check boundaries, persistence and keyboard behavior without models |
| Offline benchmark | Run scripted contracts, three repetitions per case | Check orchestration and security outcomes, streaming TTFT and repeat success-rate metrics |
| Build | Produce wheel and source distribution | Verify packaging metadata and included resources |
| Installed wheel smoke | Install wheel into a fresh environment, change to `/tmp`, run both entry points and check resources/dependencies | Catch imports that work only from the checkout and missing package data |
| Upload reports | Save JSON reports separately for each Python version, even after earlier failure when available | Make traces and failures inspectable |

The workflow grants only read access to repository contents. Dependency installation
and artifact upload use the network; tests and benchmarks do not invoke live models
or need Ollama credentials, a GPU, or llama.cpp. The workflow does not deploy, publish
packages, or run the optional cloud judge. It cannot replace a live hardware/model
smoke test or verify terminal rendering on every terminal emulator.

**Is it required?** Harness runs without GitHub Actions. CI is a development gate
that catches regressions before merge and makes verification repeatable for other
contributors. To enforce it, configure GitHub branch protection/rulesets to require
the workflow checks; merely adding this YAML does not prevent merging failures.
Equivalent checks can run in another CI service if the project moves elsewhere.

## Cleanup decisions

The restructuring removed:

- Nine obsolete `src.*` forwarding modules, including the stale `2.1.0` version.
  External integrations must import `harness.*` instead.
- The old golden-path datasets, CSV results, radar chart and plotting script.
  Their approval/scoring conventions differed from current contracts. Historical
  files remain recoverable from Git, rather than shipping as current benchmarks.
- Hidden `/inspect` and `/forget` runtime handlers. Full details and JSON export
  cover inspection; confirmed browser deletion calls `AgentRuntime.forget_memory`
  directly. Plain-mode deletion uses `harness memory forget ID`.
- Session cards' dependency on the memory-specific card class. Both now use a
  small generic `browser.CardControl` for keyboard focus and mouse scrolling.
- Filename-based evaluation track guessing; custom datasets label their track.

The audit retained code with active consumers: plain streaming output, synchronous
extraction for programmatic/offline runtimes, local CPU extraction as a configurable
alternative, old llama.cpp token-count endpoints, and the two legacy configuration
key aliases. These serve current workflows or compatibility with saved settings.
Dynamically registered tool methods and key-binding callbacks are active even when
an AST scan finds no direct call by name.

Ignored local logs, prior Qdrant data, credentials, personal notes, virtual
environments and caches are not obsolete source code. The cleanup does not delete
that data. In particular, old checkout `execution_traces/` files are separate from
current `<state_dir>/audit/`; inspect and remove them manually if no longer wanted.
