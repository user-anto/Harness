"""Portable console entry point for Linux and WSL."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from . import __version__
from .config import load_settings
from .commands import COMMANDS, command_help


def parser():
    result = argparse.ArgumentParser(prog="harness", description="Local coding and research assistant. Approved shell commands run on the host, unsandboxed.")
    result.add_argument("--version", action="version", version=__version__)
    result.add_argument("--workspace", type=Path, default=Path.cwd())
    result.add_argument("--config", type=Path, help="YAML config; default $XDG_CONFIG_HOME/harness/config.yaml")
    result.add_argument("--state-dir", type=Path, help="Override private runtime storage")
    result.add_argument("--no-memory", action="store_true", help="Disable memory reads, writes and extraction for this session")
    result.add_argument("--session", help="Explicit persistent session ID; reuse to resume. Omit for a fresh session.")
    result.add_argument("--prompt", help="Process one request and exit")
    result.add_argument("--plain", action="store_true", help="Use a plain input prompt without the terminal UI; responses still stream")
    result.add_argument("command", nargs="?", choices=["chat", "doctor", "memory", "sessions"], default="chat")
    result.add_argument("memory_action", nargs="?", choices=["list", "forget", "export"])
    result.add_argument("memory_id", nargs="?")
    result.add_argument("--probe", action="store_true", help="Doctor: check an already running model server and token counting")
    return result


def doctor(settings, probe=False):
    binary = os.getenv("HARNESS_LLAMA_SERVER_BIN") or shutil.which("llama-server")
    bundled = Path("/usr/local/lib/ollama/llama-server")
    if not binary and bundled.is_file() and os.access(bundled, os.X_OK):
        binary = str(bundled)
    report = {"python": sys.version.split()[0], "workspace": str(settings.workspace),
              "state_dir": str(settings.state_dir), "model_file": str(Path(settings.model).expanduser()),
              "model_exists": bool(settings.model) and Path(settings.model).expanduser().is_file(),
              "llama_server": binary, "ollama": shutil.which("ollama"),
              "input_budget": settings.input_budget, "memory_enabled": settings.memory_enabled}
    from .credentials import ollama_api_key
    report["web_search_key_available"] = bool(ollama_api_key(settings))
    if probe:
        from .backend import LlamaBackend, ServerManager
        from .tools import ToolRegistry
        manager = ServerManager(settings)
        import httpx
        with httpx.Client(base_url=settings.server_url, timeout=5, trust_env=False) as client:
            manager._verify(client)
        backend = LlamaBackend(manager.settings)
        try:
            count = backend.count_tokens({"model": Path(settings.model).name,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": ToolRegistry(settings).schemas, "max_tokens": settings.output_tokens})
            report.update(probe_input_tokens=count, server_context=manager.settings.model_context)
        finally:
            backend.close()
    print(json.dumps(report, indent=2))
    return 0 if report["model_exists"] and binary and (not settings.memory_enabled or report["ollama"]) else 1


def open_memory(settings):
    import ollama
    from .memory import MemoryStore
    client = ollama.Client(timeout=30)

    def embed(text):
        return client.embed(model=settings.embedding_model, input=text).embeddings[0]

    return MemoryStore(settings.state_dir / "memory", embed, embedding_model=settings.embedding_model)


def chat(args, settings, ui):
    from .backend import LlamaBackend, ServerManager
    from .runtime import AgentRuntime
    manager = ServerManager(settings)
    if args.session is None:
        from .sessions import new_session_id
        args.session = new_session_id()
    backend = memory = runtime = None
    try:
        ui.banner()
        ui.status("startup")
        settings = manager.start()
        backend = LlamaBackend(settings)
        backend.on_reasoning = ui.reasoning if settings.show_reasoning else None
        memory = open_memory(settings) if settings.memory_enabled else None
        runtime = AgentRuntime(settings, backend, memory=memory, approve=ui.approve,
                               on_text=ui.write, on_status=ui.status, on_answer=ui.answer)
        if memory and settings.memory_background:
            from .curation import MemoryCurator
            runtime.curator = MemoryCurator(settings, memory, audit=runtime.audit, on_activity=ui.background)
        ui.message(f"Harness {__version__} | workspace {settings.workspace} | session {args.session}")
        previous = runtime.snapshot(args.session)
        if previous.get("messages"):
            ui.message(f"Resumed saved session: {len(previous['messages'])} retained messages. /clear starts fresh here.")
        else:
            ui.message("New session. Global memories may still be recalled.")
        ui.message(f"GPU layers (requested): {settings.gpu_layers} | input budget: {settings.input_budget:,} tokens")
        if memory and settings.memory_background:
            ui.message(f"Memory curation: {settings.memory_model} | keep-alive requested: {settings.memory_keep_alive}")
        if settings.gpu_layers == 0:
            ui.message("CPU execution selected (gpu_layers: 0). Set gpu_layers: auto to enable GPU fitting.")
        ui.message("Approved shell commands run on the host, unsandboxed. Each write and command requires its own approval.")
        ui.message("Commands: " + ", ".join(command.name for command in COMMANDS))
        while True:
            try:
                prompt = args.prompt if args.prompt is not None else ui.read_prompt()
            except EOFError:
                break
            if prompt.strip().lower() == "/bye":
                break
            if not prompt.strip():
                if args.prompt is not None:
                    break
                continue
            try:
                ui.begin_turn()
                command = next((item for item in COMMANDS if item.name == prompt.strip()), None)
                if (prompt.strip() == "/memories" or prompt.startswith("/memories ")) and hasattr(ui, "browse_memories"):
                    ui.browse_memories(runtime, args.session, prompt.partition(" ")[2].strip())
                elif prompt.strip() == "/sessions" and hasattr(ui, "browse_sessions"):
                    args.session = ui.browse_sessions(runtime, args.session)
                elif prompt.strip() == "/help":
                    ui.message(command_help())
                elif command and command.arguments:
                    ui.message(f"Usage: {command.name} {command.arguments}")
                else:
                    answer = runtime.plan(prompt[6:], args.session) if prompt.startswith("/plan ") else runtime.turn(prompt, args.session)
                    if runtime.memory_view is not None:
                        ui.memories(*runtime.memory_view)
                    else:
                        ui.answer(answer)
                    for warning in runtime.warnings:
                        ui.message(warning)
            except Exception as exc:
                ui.close()
                ui.message(f"Request failed: {exc}")
                if args.prompt is not None:
                    return 1
            if args.prompt is not None:
                break
        return 0
    finally:
        ui.close()
        if runtime:
            runtime.close()
        if memory:
            memory.close()
        if backend:
            backend.close()
        manager.close()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        settings = load_settings(args.workspace, args.config, args.state_dir, args.no_memory)
        if args.command == "sessions":
            from .sessions import list_sessions, render_sessions
            print(render_sessions(list_sessions(settings.state_dir),
                                  color=sys.stdout.isatty() and "NO_COLOR" not in os.environ))
            return 0
        if args.command == "doctor":
            return doctor(settings, args.probe)
        if args.command == "memory":
            if not settings.memory_enabled:
                raise ValueError("Memory is disabled")
            memory = open_memory(settings)
            try:
                if args.memory_action == "export":
                    print(json.dumps(memory.records(), indent=2))
                elif args.memory_action in {"list", None}:
                    from .memory_view import render_memories
                    print(render_memories(memory.records(), color=sys.stdout.isatty() and "NO_COLOR" not in os.environ,
                                          width=shutil.get_terminal_size().columns))
                elif args.memory_action == "forget":
                    if not args.memory_id:
                        raise ValueError("Provide a memory ID")
                    deleted = memory.forget(args.memory_id)
                    if deleted:
                        from .curation import discard_pending
                        discard_pending(settings.state_dir)
                    print("Memory forgotten." if deleted else "Memory not found.")
            finally:
                memory.close()
            return 0
        from .ui import TerminalUI
        if sys.stdout.isatty() and sys.stdin.isatty() and not args.plain and args.prompt is None:
            from .tui import ChatUI
            ui = ChatUI()
            return ui.run(lambda: chat(args, settings, ui))
        return chat(args, settings, TerminalUI(interactive=False))
    except KeyboardInterrupt:
        print("\nInterrupted. Inspect any pending host command's effects before retrying.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Harness: {exc}", file=sys.stderr)
        return 1
