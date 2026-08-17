import json
import os
import time
import subprocess
import atexit
import urllib.request
from urllib.error import URLError
import yaml
from typing import Annotated, Literal
from typing_extensions import TypedDict, NotRequired
import re
import base64
import binascii
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolNode
from .cli import render_terminal_ui, Spinner
from .utils import start_llama_server, _convert_messages_to_openai, _stream_llm
from .llm import client, guard_llm
from .prompts import SYSTEM_PROMPT, GUARD_PROMPT, get_planner_prompt, get_planning_mode_prompt
from .audit import audit_logger
from .tools import tools
import uuid
import threading
from .memory import retrieve_context, embed_and_save

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load configuration and initialize OpenAI client
with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

MODEL_CTX = config.get("ctx-size") * 1024
orch_model = config.get("default_orch_model")

llama_process = start_llama_server(orch_model)

tool_node = ToolNode(tools)
openai_tools = [convert_to_openai_tool(t) for t in tools]

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    planning_mode: NotRequired[bool]
    plan_tasks: NotRequired[list[str]]
    completed_tasks: NotRequired[list[str]]

def deterministic_guard(prompt: str) -> bool:
    if len(prompt) > 4000:
        return False
        
    lower_prompt = prompt.lower()
    blacklist = [
        "ignore previous instructions",
        "ignore all previous",
        "system prompt",
        "initial instructions",
        "forget everything",
        "you are now",
        "sudo",
        "bypass"
    ]
    if any(phrase in lower_prompt for phrase in blacklist):
        return False
        
    # Regex (Base64) check for long payloads
    if re.search(r'\b[A-Za-z0-9+/=]{40,}\b', prompt):
        return False
        
    # Regex (Hex) check for long payloads
    if re.search(r'\b(?:[0-9a-fA-F]{2}){20,}\b', prompt):
        return False
        
    # Inline decode and check for shorter obfuscated payloads
    for word in prompt.split():
        # Check Base64
        clean_b64 = re.sub(r'[^A-Za-z0-9+/=]', '', word)
        if len(clean_b64) >= 16 and len(clean_b64) % 4 == 0:
            try:
                decoded = base64.b64decode(clean_b64).decode('utf-8').lower()
                if any(phrase in decoded for phrase in blacklist):
                    return False
            except Exception:
                pass
                
        # Check Hex
        clean_hex = re.sub(r'[^A-Fa-f0-9]', '', word)
        if len(clean_hex) >= 16 and len(clean_hex) % 2 == 0:
            try:
                decoded = binascii.unhexlify(clean_hex).decode('utf-8').lower()
                if any(phrase in decoded for phrase in blacklist):
                    return False
            except Exception:
                pass
        
    # Entropy / Special character ratio check
    non_spaces = len(prompt.replace(" ", ""))
    if non_spaces > 20: 
        # Exclude math symbols + - * / ( ) = . from the special character count
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s\+\-\*\/\(\)\=\.]', prompt))
        if (special_chars / non_spaces) > 0.3:
            return False
            
    return True

def guard_node(prompt: str) -> bool:
    if not deterministic_guard(prompt):
        return False
        
    response = guard_llm.invoke([
        SystemMessage(content=GUARD_PROMPT),
        HumanMessage(content=prompt)
    ])
    if "YES" in str(response.content).upper():
        return False
    return True

def input_node(state: State):
    out = {"messages": []}
    
    if "planning_mode" not in state:
        out["planning_mode"] = False
            
    # 2. Get input and run guardrail
    user_input = input("\n\033[32m> ")
    print("\033[0m> ", end="", flush=True)
    prompt_id = audit_logger.log_prompt(user_input)
    
    user_msg = HumanMessage(content=user_input, id=prompt_id)
    
    if user_input.strip().lower() == '/bye':
        out["messages"].append(user_msg)
        return out
        
    is_safe = guard_node(user_input)
    if not is_safe:
        print("\033[91m[Guardrail] Prompt injection detected. Request denied.\033[0m")
        audit_logger.log_error("Prompt injection detected")
        reject_msg = AIMessage(content="Prompt injection detected. Request denied.", id=str(uuid.uuid4()))
        out["messages"].extend([user_msg, reject_msg])
        return out
        
    if not state["messages"]:
        out["messages"].extend([SYSTEM_PROMPT, user_msg])
    else:
        out["messages"].append(user_msg)
        
    return out

def planner_node(state: State):
    last_msg = state["messages"][-1].content
    plan_query = last_msg.replace("/plan", "").strip()
    if not plan_query:
        plan_query = "Create a plan for the next tasks."
        
    spinner_text = "Generating plan..."
    
    while True:
        plan_prompt = get_planner_prompt(plan_query)
        
        with Spinner(spinner_text):
            response = client.chat.completions.create(
                model=orch_model,
                messages=[{"role": "system", "content": plan_prompt}],
                temperature=0.2
            )
            
        plan_text = response.choices[0].message.content.strip()
        
        tasks_file = os.path.join(os.getcwd(), "agent_tasks.art.md")
        with open(tasks_file, "w") as f:
            f.write(plan_text + "\n")
            
        subprocess.run(["glow", tasks_file])
        
        user_choice = input("\n\033[36mProceed with plan? (Y/n/feedback): \033[0m").strip()
        
        if user_choice.lower() in ["y", ""]:
            tasks = [ln.strip().strip('"').split(":", 1)[0].strip()
                     for ln in plan_text.splitlines() if ln.strip()]
            return {"planning_mode": True, "plan_tasks": tasks, "completed_tasks": []}
        elif user_choice.lower() == "n":
            cancel_msg = AIMessage(content="Plan generation cancelled by user.", id=str(uuid.uuid4()))
            return {"planning_mode": False, "messages": [cancel_msg]}
        else:
            plan_query = f"Original request: {plan_query}\nUser feedback: {user_choice}"
            spinner_text = "Improving plan..."

def orchestrator_node(state: State):
    out = {}
    
    if state.get("planning_mode"):
        if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
            last_tool_msg = state["messages"][-1]
            if last_tool_msg.name == "task_complete":
                pending = list(state.get("plan_tasks", []))
                done = list(state.get("completed_tasks", []))
                if pending:
                    popped_task = pending.pop(0)
                    done.append(popped_task)
                    out["plan_tasks"] = pending
                    out["completed_tasks"] = done
                    state["plan_tasks"] = pending
                    state["completed_tasks"] = done
                
                if not pending:
                    state["planning_mode"] = False
                    out["planning_mode"] = False
    
    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        tool_msgs = []
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                tool_msgs.insert(0, msg)
            else:
                break
        for msg in tool_msgs:
            audit_logger.log_tool_output(msg.name, msg.content)
            
    llm_messages = list(state["messages"])
    
    if state.get("planning_mode"):
        pending = state.get("plan_tasks", [])
        done = state.get("completed_tasks", [])
        plan_sys_msg = SystemMessage(
            content=get_planning_mode_prompt(pending, done),
            id=str(uuid.uuid4()),
        )
        if len(llm_messages) > 1:
            llm_messages = llm_messages[:1] + [plan_sys_msg] + llm_messages[1:]
        else:
            llm_messages.append(plan_sys_msg)
            
    response, usage = _stream_llm(llm_messages)
    out["messages"] = [response]
    
    if usage:
        out["prompt_tokens"] = usage.prompt_tokens
        out["response_tokens"] = usage.completion_tokens
        out["total_tokens"] = usage.total_tokens
        
    return out

def archival_node(state: State):
    out = {"messages": []}
    
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
        last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if last_human and last_human.content.strip().lower() != '/bye':
            trace_id = os.path.basename(audit_logger.log_file)
            threading.Thread(
                target=embed_and_save,
                args=(last_human.content, last_message.content, trace_id, last_human.id, last_message.id)
            ).start()
            
    window_limit = config.get("sliding_window_tokens", 96) * 1024
    # ponytail: atomic turn-block removal — never leave dangling ToolMessages
    # without their AIMessage. Upgrade: keep most-recent SystemMessage if budget allows.
    estimated_tokens = state.get("total_tokens", 0)
    messages = state["messages"]

    while estimated_tokens > window_limit and len(messages) > 1:
        block = []
        # skip leading non-Human messages (e.g. injected SystemMessages)
        i = next((k for k, m in enumerate(messages) if isinstance(m, HumanMessage) and k > 0), None)
        if i is None or i >= len(messages) - 1:
            break
        j = i
        if isinstance(messages[j + 1], AIMessage):
            block.append(messages[j])
            block.append(messages[j + 1])
            j += 2
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                block.append(messages[j])
                j += 1
        else:
            block.append(messages[i])
            j = i + 1

        for m in block:
            out["messages"].append(RemoveMessage(id=m.id))
        estimated_tokens -= sum(len(str(m.content)) for m in block) // 4
        messages = [m for m in messages if m.id not in {b.id for b in block}]

    if out["messages"]:
        out["total_tokens"] = estimated_tokens

    return out

def route_after_input(state: State) -> Literal["orchestrator_node", "input_node", "planner_node", END]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and "Prompt injection detected" in str(last_message.content):
        return "input_node"
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if last_human:
        if last_human.content.strip().lower() == '/bye':
            # Clean up all files ending with .art.md in the working directory
            try:
                for filename in os.listdir(os.getcwd()):
                    if filename.endswith(".art.md"):
                        file_path = os.path.join(os.getcwd(), filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            except Exception:
                pass
            return END
        if re.search(r"/plan", last_human.content):
            return "planner_node"
    return "orchestrator_node"

def should_continue(state: State) -> Literal["tool_node", "input_node"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return "input_node"

# Graph structure
workflow = StateGraph(State)
workflow.add_node("input_node", input_node)
workflow.add_node("planner_node", planner_node)
workflow.add_node("orchestrator_node", orchestrator_node)
workflow.add_node("archival_node", archival_node)
workflow.add_node("tool_node", tool_node)

# Set up edges
workflow.add_edge(START, "input_node")
workflow.add_conditional_edges("input_node", route_after_input)
workflow.add_edge("planner_node", "orchestrator_node")
workflow.add_edge("orchestrator_node", "archival_node")
workflow.add_conditional_edges("archival_node", should_continue)
workflow.add_edge("tool_node", "orchestrator_node")

# Compile state graph application
app = workflow.compile()

if __name__ == "__main__":
    audit_logger.initialize()
    
    try:
        with open(os.path.join(BASE_DIR, "graph_visualization.png"), "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
    except Exception:
        pass

    render_terminal_ui()

    response = app.invoke({})