# WSL smoke test

Run inside Ubuntu/WSL2 with the checkout on the Linux filesystem:

```bash
cd ~/projects/Harness
make install
make test
make lint
make benchmark
make build
.venv/bin/harness --config config.yaml --workspace "$PWD" doctor
```

For a live smoke test, first set a GGUF path and a context size appropriate for your
hardware in an explicitly selected config. Start a new session with a temporary
workspace and state directory so your normal memories are separate:

```bash
smoke_workspace=$(mktemp -d)
smoke_state=$(mktemp -d)
.venv/bin/harness --config "$PWD/config.yaml" --workspace "$smoke_workspace" --state-dir "$smoke_state" --session smoke
```

1. Ask for `2+2`; verify the answer and successful math tool result.
2. Ask to write a file; press Enter and confirm no file exists.
3. Repeat, type `yes`, and verify the file contents.
4. Ask to run `printf hello`; verify the exact host command/cwd preview; approve once.
5. `/remember I prefer concise answers`, exit, restart with the same paths/session,
   and ask the preference. Inspect `/memories` and its source/evidence.
6. Open `/sessions`, search by title, load another session with Enter, and confirm
   its history and workspace. Check Esc returns without switching. Delete a disposable
   session only after confirming `yes`; deleting the current one starts a fresh chat.
7. In `/memories`, test search, confirmed deletion and Alt+M editing on a disposable
   fact. Verify the ID stays fixed after an edit and `/inspect` and `/forget` are absent
   from the command menu.
8. Try a read of `/etc/passwd` and confirm the path refusal.
9. Exit and confirm an owned llama-server process terminated. An explicitly reused
   server should stay running. Keep temporary paths for inspection or remove them manually.

From Windows tools, invoke Linux Git through WSL rather than Windows Git over the
UNC filesystem: `wsl.exe --distribution Ubuntu --cd /home/ranta_unix/projects/Harness --exec git status`.
