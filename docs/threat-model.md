# Threat model

Harness assumes one trusted local user, an owned state directory, and a loopback
model server. Repository files, pages, fetched snippets, tool outputs, summaries,
and retrieved memories may contain malicious or false text.
Background memory curation uses Ollama Cloud by default, sending bounded eligible
user excerpts and redacted final assistant replies to that external service. Raw
tool calls/results and reasoning are excluded. Secret filtering is best effort;
arbitrary sensitive prose may still be sent. Use `--no-memory` or select a local
`memory_model` to avoid cloud curation. Stored memories and embeddings remain local.

File/search tools resolve their paths under the selected workspace and reject known
credential locations and Git metadata. Symlink traversal outside that root is refused.
These checks do not sandbox the Python process or protect against another process
changing the filesystem concurrently. Run untrusted executable repositories in an OS
sandbox or disposable machine when that stronger boundary is needed.

Host shell commands have no filesystem or network confinement. Each exact command
and cwd needs a fresh `yes`; approvals are never inferred from text in model context.
Output/time limits are resource controls, not a sandbox. A model can still propose a
harmful command, so the human decision matters. Read-only Git tools disable external
diff helpers; repositories and host Git configuration should still be treated carefully.

HTTP fetches only allow HTTP(S), validate every DNS result and redirect, reject
non-public targets, and connect to a checked IP. HTTPS retains certificate and SNI
validation for the original host. Responses have byte/time/redirect limits. Web search
uses the external Ollama service only when configured; do not send sensitive queries.
The web-search adapter reads only `OLLAMA_API_KEY` from the workspace `.env` or an
explicitly configured `env_file`, with exported environment values taking precedence.
It does not evaluate shell syntax or import other settings from that file. Credential
values are used for the Ollama authorization header, not passed into model context.

Prompt wording and labels assist the model but do not guarantee injection resistance.
The enforceable boundary is tool policy and human approval. Model refusals and secret
system-prompt concealment are not security metrics.

Known secret patterns are redacted before checkpoints/tool results and refused by
memory. Detection cannot recognize every arbitrary secret. Avoid providing credentials
to the assistant. Data is stored with 0700 directories and 0600 files, without encryption.
Global memories can cross projects. Inferences can be incorrect and must remain labeled.
Historical assistant answer notes can also be wrong or stale. They carry dates,
unverified status and revalidation markers; optional related URLs do not prove the
claim. These prompt labels reduce ambiguity but cannot guarantee model compliance.
Confirmed deletion in the Memories tab deletes versions from curated memory and
clears the active session context;
other sessions and historical checkpoints expire through retention rather than a
global semantic erasure. To erase all session data, stop Harness and remove the
specific private state directory after preserving any wanted backups.

Audit logs are mutable, best-effort local records. They are not non-repudiation evidence.
Managed subprocess shutdown cannot survive machine power loss or every external kill.


Loading a session through the Sessions tab also restores its workspace for file and
shell tools; choosing a session does not reload configuration from that repository.
The workspace must still exist. Session deletion removes its checkpoint records and
display title, but does not erase global memories or separate audit files. Pending
curation is invalidated on deletion to prevent a late commit from deleted context.


## Security evaluation boundaries

`make security-eval` runs scripted enforcement cases; `--live --suite security`
adds model-behavior probes. They cover file/page injection, malicious tool output,
forged or reused approvals, destructive/upload command requests, and attempts to
execute code through arithmetic. Injection cases require successful content exposure
and reject injected write/command attempts in live mode, even when approval prevents
the side effect. Direct unsafe requests may be refused or denied at approval.

The runner uses temporary workspaces, fixture pages and blocked external web search.
Shell approvals are denied; an independent process guard prevents host command
execution and records a test failure if dispatch is reached. These finite probes
cannot prove general injection resistance or safety after a human approves a harmful
host command. See [evaluation details](../evals/README.md).
