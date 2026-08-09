import os
import json
import time
import subprocess
import atexit
import urllib.request
import sys
import threading
import uuid
from urllib.error import URLError
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from audit import audit_logger
from tools import tools
from llm import client, orch_model, MODEL_CTX

from cli import Spinner

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
openai_tools = [convert_to_openai_tool(t) for t in tools]

# llama-server initialization helper
def start_llama_server(model_path):
    model_path = os.path.expanduser(model_path)
    try:
        req = urllib.request.Request("http://localhost:8000/v1/models")
        with urllib.request.urlopen(req, timeout=1):
            return None # Server is already running
    except URLError:
        pass # Server not running, proceed to start

    spinner = Spinner("Starting llama-server...")
    spinner.__enter__()
    log_file = open(os.path.join(BASE_DIR, "llama-server.log"), "w")
    cmd = [
        "llama-server",
        "-m", model_path,
        "-ngl", "999",
        "--ctx-size", str(MODEL_CTX),
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    
    def cleanup():
        with Spinner("Shutting down llama-server..."):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            log_file.close()
    
    atexit.register(cleanup)
    
    # Wait for server to be ready
    for _ in range(30):
        try:
            req = urllib.request.Request("http://localhost:8000/v1/models")
            with urllib.request.urlopen(req, timeout=1):
                spinner.__exit__(None, None, None)
                return process
        except Exception:
            time.sleep(1)
    
    spinner.__exit__(None, None, None)
    print("\033[91mWarning: llama-server did not start in time.\033[0m")
    return process

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

    if usage and not tool_calls_dict:
        print(f"\033[95m[Total Tokens: {usage.total_tokens}]\033[0m")

    full_content = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts)
    
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