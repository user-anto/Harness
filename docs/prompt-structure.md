# Main-model request structure

Harness assembles a chat-completions request, rather than concatenating everything
into one unstructured string. The llama.cpp chat template renders that request into
the model's effective prompt.

```text
messages:
  1. system: Harness instructions, current UTC time, tool/error/approval rules,
             memory trust rules, current web-search credential availability
  2. user:   <context_data trust='data-only'>   [when nonempty]
               active_task: goal, constraints, decisions, completed/pending work,
                            evidence references
               older_turn_excerpts: attributed excerpts of compacted history
               memories: selected global memory records, with provenance;
                         answer notes include dates and revalidation markers
             </context_data>
  3. retained conversation in chronological order:
       user → assistant → tool results → assistant → …
       newest user request
       assistant tool calls → matching tool results   [during the current loop]

tools:       available function names, descriptions, and JSON argument schemas
max_tokens:  output-token reserve (4,096 by default)
model:       configured local model
```

Answer notes record prior assistant conclusions and are not verified knowledge.
System instructions require rechecking them before factual reuse, especially for
time-sensitive claims. Source URLs attached to notes describe captured context,
not independent verification.

The context-data block has a `user` role on the wire; it is explicitly labeled as
data, and the system instructions prohibit treating retrieved text as new policy
or approval. Tools are a separate API field whose rendering is template dependent.
After each tool execution, the next generation sees its matching tool result with
status, provenance and truncation metadata. The current tool-call group stays intact.

The exact fully rendered request is counted before generation. The default input
ceiling is 32,768 tokens, reduced when needed for model capacity, output reserve and
headroom. Older whole turns may become bounded attributed excerpts. At most five
memories are selected within a 1,000-token retrieval budget. Ordinary file contents
enter through file/search tools; Harness does not automatically paste the repository
or `.env` into the prompt. API-key values never enter this credential-status field.

A fresh CLI launch starts with no earlier conversation messages. Global memory
recall is independent of that history. `--session ID` or Enter on a saved Sessions card explicitly restores messages,
task state and excerpts from SQLite, which is why a resumed session can recognize
a repeated question even when `/memories` is empty. The model does not independently
read SQLite or remember prior HTTP requests.

Memory curation is a separate request to the Ollama Cloud extractor, including the
user excerpt and final assistant answer as context, with raw tool messages excluded. That extraction
prompt and its candidate JSON are not appended to the main model's conversation.
