import os
import json
import time
import subprocess
import atexit
import urllib.request
import sys
import threading
import uuid
import shutil
from urllib.error import URLError
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from .audit import audit_logger
from .tools import tools
from .llm import client, orch_model, MODEL_CTX

from .cli import Spinner

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
openai_tools = [convert_to_openai_tool(t) for t in tools]

# llama-server initialization helper
def start_llama_server(model_path):
    model_path = os.path.expanduser(model_path)
    server_url = os.getenv("HARNESS_LLAMA_SERVER_URL", "http://localhost:8000")

    try:
        req = urllib.request.Request(f"{server_url}/v1/models")
        with urllib.request.urlopen(req, timeout=1):
            return None # Server is already running
    except URLError:
        pass # Server not running, proceed to start

    spinner = Spinner("Starting llama-server...")
    spinner.__enter__()
    log_path = os.path.join(BASE_DIR, "llama-server.log")
    timeout = float(os.getenv("HARNESS_LLAMA_STARTUP_TIMEOUT", "120"))

    # A locally built CUDA server can be first on PATH even when the CUDA
    # runtime is not installed. Ollama ships a CPU-capable llama-server, so
    # keep it as a fallback instead of allowing a dead process to reach the
    # first LLM request.
    override = os.getenv("HARNESS_LLAMA_SERVER_BIN")
    candidates = []
    if override:
        candidates.append((override, False))
    else:
        path_server = shutil.which("llama-server")
        if path_server:
            candidates.append((path_server, False))
        bundled_server = "/usr/local/lib/ollama/llama-server"
        if (os.path.isfile(bundled_server)
                and os.access(bundled_server, os.X_OK)
                and os.path.realpath(path_server or "") != os.path.realpath(bundled_server)):
            candidates.append((bundled_server, True))

    if not candidates:
        spinner.__exit__(None, None, None)
        raise RuntimeError(
            "llama-server was not found. Install llama.cpp or set "
            "HARNESS_LLAMA_SERVER_BIN to its executable path."
        )

    failures = []
    for server_bin, cpu_only in candidates:
        # The bundled Ollama server is usable without CUDA. Explicitly avoid
        # requesting GPU layers for it; this also makes the fallback portable.
        gpu_layers = "0" if cpu_only else "999"
        cmd = [
            server_bin,
            "-m", model_path,
            "-ngl", gpu_layers,
            "--ctx-size", str(MODEL_CTX),
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]
        log_mode = "w" if not failures else "a"
        log_file = open(log_path, log_mode)
        log_file.write(f"Starting: {' '.join(cmd)}\n")
        log_file.flush()

        try:
            launch_env = None
            cuda12_dir = "/usr/local/lib/ollama/cuda_v12"
            if not cpu_only and os.path.isdir(cuda12_dir):
                # The local llama.cpp build is linked against CUDA 12, while
                # this host's default CUDA installation is version 13.
                # Ollama already provides the compatible runtime libraries.
                launch_env = os.environ.copy()
                old_library_path = launch_env.get("LD_LIBRARY_PATH", "")
                launch_env["LD_LIBRARY_PATH"] = os.pathsep.join(
                    part for part in (cuda12_dir, old_library_path) if part
                )
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=launch_env,
            )
        except OSError as exc:
            log_file.close()
            failures.append(f"{server_bin}: {exc}")
            continue

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                req = urllib.request.Request(f"{server_url}/v1/models")
                with urllib.request.urlopen(req, timeout=1):
                    spinner.__exit__(None, None, None)

                    def cleanup(process=process, log_file=log_file):
                        with Spinner("Shutting down llama-server..."):
                            if process.poll() is None:
                                process.terminate()
                                try:
                                    process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                            log_file.close()

                    atexit.register(cleanup)
                    return process
            except Exception:
                if process.poll() is not None:
                    break
                time.sleep(1)

        return_code = process.poll()
        if return_code is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            failures.append(f"{server_bin}: timed out after {timeout:g}s")
        else:
            failures.append(f"{server_bin}: exited with code {return_code}")
        log_file.close()

    spinner.__exit__(None, None, None)
    try:
        with open(log_path, "r") as log_file:
            log_tail = "".join(log_file.readlines()[-12:]).strip()
    except OSError:
        log_tail = "(server log unavailable)"
    raise RuntimeError(
        "llama-server could not be started. "
        + "; ".join(failures)
        + f"\nSee {log_path} for details.\n{log_tail}"
    )

# Convert LangChain message history to OpenAI message schema format
def _convert_messages_to_openai(messages):
    openai_msgs = []
    for msg in messages:
        if isinstance(msg, tuple):
            role, content = msg
            openai_msgs.append({"role": role, "content": content})
        elif msg.type == "human":
            openai_msgs.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            d = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    }
                    for tc in msg.tool_calls
                ]
            openai_msgs.append(d)
        elif msg.type == "tool":
            openai_msgs.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content
            })
        elif msg.type == "system":
            openai_msgs.append({"role": "system", "content": msg.content})
    return openai_msgs

# LLM streaming response helper function bypassing LangChain
def _stream_llm(messages):
    openai_msgs = _convert_messages_to_openai(messages)
    response = client.chat.completions.create(
        model=orch_model,
        messages=openai_msgs,
        tools=openai_tools,
        stream=True,
        stream_options={"include_usage": True}
    )

    reasoning_parts = []
    content_parts = []
    tool_calls_dict = {}
    usage = None

    in_reasoning = False

    try:
        for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 1. Extract and stream reasoning content in grey color and italics
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                if not in_reasoning:
                    print("\033[90m\033[3mThinking: ", end="", flush=True)
                    in_reasoning = True
                print(reasoning, end="", flush=True)
                reasoning_parts.append(reasoning)

            # 2. Extract and stream standard text content in white/default color
            content = getattr(delta, "content", None)
            if content:
                if in_reasoning:
                    print("\033[0m\n", end="", flush=True)
                    in_reasoning = False
                print(content, end="", flush=True)
                content_parts.append(content)

            # 3. Extract tool calls
            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                if in_reasoning:
                    print("\033[0m\n", end="", flush=True)
                    in_reasoning = False
                for tc in tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_dict:
                        tool_calls_dict[idx] = {
                            "id": tc.id,
                            "name": tc.function.name if tc.function else "",
                            "arguments": tc.function.arguments if tc.function else ""
                        }
                    else:
                        if tc.id:
                            tool_calls_dict[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_dict[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_dict[idx]["arguments"] += tc.function.arguments
    except KeyboardInterrupt:
        if in_reasoning:
            print("\033[0m", end="")
        print("\n\033[93m[Generation interrupted by user (Ctrl+C)]\033[0m")


    if in_reasoning:
        print("\033[0m")
    else:
        print()

    full_content = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts)
    
    if usage:
        approx_reasoning_tokens = len(full_reasoning) // 4
        try:
            usage.completion_tokens = max(0, usage.completion_tokens - approx_reasoning_tokens)
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        except Exception:
            class UsageMock:
                def __init__(self, p, c):
                    self.prompt_tokens = p
                    self.completion_tokens = c
                    self.total_tokens = p + c
            usage = UsageMock(usage.prompt_tokens, max(0, usage.completion_tokens - approx_reasoning_tokens))
            
    if usage and not tool_calls_dict:
        print(f"\033[95m[Total Tokens: {usage.total_tokens}]\033[0m")
    
    resp_id = str(uuid.uuid4())
    
    if full_reasoning:
        audit_logger.log_reasoning(full_reasoning)

    # Reconstruct tool calls for LangGraph state update
    parsed_tool_calls = []
    for idx, tc in tool_calls_dict.items():
        try:
            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
        except json.JSONDecodeError:
            args = {}
        
        audit_logger.log_tool_call(tc["name"], args)
        
        parsed_tool_calls.append({
            "name": tc["name"],
            "args": args,
            "id": tc["id"],
            "type": "tool_call"
        })
        
    if full_content:
        resp_id = audit_logger.log_final_response(full_content)

    # Return AIMessage to update LangGraph state
    ai_msg = AIMessage(
        content=full_content,
        tool_calls=parsed_tool_calls,
        id=resp_id
    )
    return ai_msg, usage
