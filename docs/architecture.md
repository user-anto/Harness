# Architecture decisions

1. **Explicit composition.** `harness.cli` loads validated settings and constructs the
   server, backend, curated store, tool registry and runtime. Module imports are inert.
   `harness/` is the only application package; obsolete `src.*` transition imports
   and the v2 global application/tool lists have been removed.
2. **Context is a view, not the transcript.** `ContextAssembler` counts a complete
   request before generation. The active user/tool group is indivisible. The latest
   user request stays at the correct chronological position so tool protocols remain
   valid. A separate data-only context message holds the task record, attributed
   summary, and memories. Metadata counters are diagnostic and do not sum exactly to
   the full-template count, which includes schema/template overhead.
3. **Extractive compaction first.** Older turns become bounded, attributed excerpts.
   This avoids an extra model claiming new facts during compaction. It can omit
   details; the task ledger and persistent checkpoints provide recovery references.
   Better semantic summaries require benchmark evidence before replacing this default.
   The `record_task` tool pins constraints and decisions only when an exact quote can
   be matched to a user turn; model-generated or tool-generated requirements are refused.
4. **Global curated memory.** Qdrant stores records and vectors together, separately
   from audit traces. Source workspace is provenance/ranking metadata, not a namespace.
   Explicit records outrank automatic claims. The extractor sees user excerpts and
   the final assistant reply. User facts require user text; dated answer notes quote
   the assistant and retain unverified historical status, even without sources. Exact
   quote/turn validation precedes any save. Confidence is not calibrated truth.
   Known explicit slots and stable extractor keys supersede conflicts; arbitrary prose
   does not have a complete semantic contradiction guarantee.
   The interactive CLI injects a bounded, durable `MemoryCurator` worker using local
   Ollama's `gpt-oss:20b-cloud` through the local API, independently of the main
   llama-server, requesting a ten-minute keep-alive. Cloud failures automatically try
   the CPU-only `llama3.2:3b` fallback with the same validation/evidence policy.
   A 60-second cooldown prevents every queued job from waiting on a failed cloud
   request, and the next eligible job after cooldown retries the cloud. Qdrant calls
   are serialized; embedding requests run outside the database lock. Commit guards
   prevent invalidated jobs from restoring forgotten memories. Explicit saves stay
   synchronous. Parsed extraction JSON and policy acceptance are distinct outcomes:
   fixed rejection codes explain rejected candidates, and a missing substantive
   answer note triggers a bounded focused retry with local fallback. This recovery
   never lowers evidence or confidence requirements and leaves accepted facts intact.
   Programmatic/evaluation runtimes without an injected curator retain
   the synchronous extraction path.
   Automatic extraction sends the newest user/final-answer pair and at most one related earlier
   excerpt; eight recent eligible turns form the local selection pool. Simple
   greetings bypass inference. The configurable 120-second timeout permits one
   background job retry before reporting failure.
   Answer notes have a separate key namespace, a 30-day recall lifetime and the
   lowest ranking priority. Related web-source metadata is attached locally, while
   raw tools remain outside the extraction request. Recall carries revalidation
   markers; related sources are not treated as proof of the note's correctness.
5. **Persistence and action recovery.** A SQLite LangGraph checkpointer persists
   single-user sessions with strict deserialization. Restart turns reconcile unfinished
   tool calls as unknown outcomes; they never automatically resume pending host actions.
   Each retry needs a fresh approval. The state directory has an exclusive runtime lock.
   CLI launches generate a new session ID unless `--session ID` explicitly selects
   a saved conversation. Resumes display the retained message count. Conversation
   checkpoints are independent of global curated memories. The Sessions tab supports
   name search, confirmed deletion and loading into the same process. Display titles
   derive from the first conversational prompt; older sessions remain `UNKNOWN`.
   Loading validates the original workspace and rebinds runtime/tool settings before
   the next request. Deleting the current session selects a fresh ID.
6. **Observable contracts.** Tool results expose status, source, timestamp and
   truncation. Context manifests expose counts, selections and omissions. Audit files
   record metadata and approvals, without reasoning traces or raw prompt transcripts.
   Checkpoints and memories still contain redacted conversation data and deserve backups.
   Model-provided reasoning can be displayed in gray italics through a separate
   callback, but is excluded from persisted messages and audit traces.

## Configuration and APIs

Public entry point: `harness [--workspace PATH] [--config PATH] [--session ID]`.
`doctor [--probe]` does not start a server. `memory list|export|forget ID` does not
require text generation. Explicit config overrides the XDG default. A repository's
config is never implicitly loaded. The credential adapter reads only
`OLLAMA_API_KEY` from an allowed `.env` path; other settings are not imported.
The two old config keys `default_orch_model`
and `ctx-size` are accepted for transition; new config uses `model` and `model_context`
in actual token units.

Programmatic composition uses `Settings`, `ContextAssembler`, `MemoryStore`,
`ToolRegistry`, `LlamaBackend` and `AgentRuntime`. Close runtime/database/backend
resources explicitly. Tool registries accept injected approval and fetch callbacks.
The backend interface provides `count_tokens(request)` and
`complete(request, on_text=None) -> (assistant_message, usage)`.

## Verification and rollout

Run tests, lint, offline contracts, wheel build, wheel smoke and doctor before use.
Compare a live benchmark report against another report with the same mode/dataset;
model weights, llama.cpp build, embedding model, hardware and configuration must also
match for a meaningful comparison. Keep both workload tracks visible. Do not publish
scripted offline results as model performance. The legacy memory is left untouched;
the new store starts empty and can be inspected/exported/deleted independently.

## Memory management interface

`MemoryBrowser` is a separate prompt-toolkit view. UI events handle selection and
local keyword filtering; the CLI worker reads snapshots and performs confirmed
deletions, so database work does not block terminal input. Search and selected IDs
survive view switches. The details pane exposes the complete record. Confirmed deletion calls
`AgentRuntime.forget_memory` directly, invalidates automatic curation and clears the
current checkpoint context; it does not synthesize a chat slash command.

Agent edits use `read_memory` and `update_memory`. The runtime binds edit authority
to the current user's `Modify memory ID:` instruction and the resolved UUID, and
revokes it at turn completion. Storage re-embeds in place and checks the record
again before commit, preserving its UUID and keeping a bounded statement history.


## Source boundaries

`browser.CardControl` shares focus, mouse navigation and scroll positioning between
Memories and Sessions; each browser owns its rendering, search and actions.
`commands.COMMANDS` is the single registry for startup hints, slash completion and
help. Removed `/inspect` and `/forget` commands are rejected locally. Full memory
JSON remains available through the details pane and `harness memory export`;
standalone deletion uses `harness memory forget ID`.

See [development and CI](development.md) for the source map and verification gates.


## Rendering and interruption

The terminal caches wrapped logical lines (including ANSI carry), search strings,
and browser layout; UIContent retrieves visible card lines rather than building
all fragments on every selection. Context diagnostic counts use a bounded
content-keyed cache; the complete request is still counted before generation.

Production main-model HTTP runs on a backend-owned asyncio loop behind the same
synchronous interface. Cancelling its active task closes the request even while
waiting for response headers. `TurnInterrupted` is recoverable in the chat loop;
a distinct per-turn event leaves the process/session shutdown flag unset. Runtime
checks before model work and each new tool call prevent scheduling more actions.
