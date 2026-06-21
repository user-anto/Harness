import json
import datetime
import yaml
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolNode
from openai import OpenAI
from tools import tools
import uuid

SUMMARIZE_AT = 48 * 1024

# Load configuration and initialize OpenAI client
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)
orch_model = config.get("default_orch_model")

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

tool_node = ToolNode(tools)
openai_tools = [convert_to_openai_tool(t) for t in tools]

# Define system prompt for the agent
current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SYSTEM_PROMPT = SystemMessage(
    content=f"""You are a terse, careful assistant.
    The current date and time is {current_date}.
    Use the smallest reasoning trace that still yields a correct answer.
    Do not explore multiple branches unless the problem is genuinely ambiguous.
    Do not restate the question. Do not show step-by-step reasoning unless the user explicitly asks for it. 
    For factual, current, niche, or uncertain questions, verify using the available web/search tools before answering.
    Do not guess or invent details. Prefer primary or official sources.
    If verification is not possible, say so clearly and answer conservatively.
    Give the final answer directly, keep it concise, and optimize for accuracy over cleverness.
    """)

SUMMARIZATION_PROMPT = """Summarize the conversation so far.
Retain all crucial facts, user preferences, unresolved questions, and important details.
The summary should be comprehensive but much more concise than the original log.
"""

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    prompt_tokens: int
    response_tokens: int
    total_tokens: int

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

    if in_reasoning:
        print("\033[0m")
    else:
        print()

    if usage and not tool_calls_dict:
        print(f"\033[95m[Total Tokens: {usage.total_tokens}]\033[0m")

    full_content = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts)

    # Reconstruct tool calls for LangGraph state update
    parsed_tool_calls = []
    for idx, tc in tool_calls_dict.items():
        try:
            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
        except json.JSONDecodeError:
            args = {}
        parsed_tool_calls.append({
            "name": tc["name"],
            "args": args,
            "id": tc["id"],
            "type": "tool_call"
        })

    # Return AIMessage to update LangGraph state
    ai_msg = AIMessage(
        content=full_content,
        tool_calls=parsed_tool_calls
    )
    return ai_msg, usage

# Orchestrator node that prompts user and manages model interaction
def orchestrator_node(state: State):
    out = {}
    
    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        # Continue streaming from LLM if returning from a tool run
        llm_messages = state["messages"]
        new_state_msgs = []
    else:
        user_input = input("\n\033[32m> ")
        print("\033[0m> ", end="", flush=True)
        user_msg = HumanMessage(content=user_input, id=str(uuid.uuid4()))
        
        # Check for /bye command
        if user_input.strip().lower() == '/bye':
            return {"messages": [user_msg]}
            
        if not state["messages"]:
            # Inject system prompt for the first turn
            llm_messages = [SYSTEM_PROMPT, user_msg]
            new_state_msgs = [SYSTEM_PROMPT, user_msg]
        else:
            # Append user message
            llm_messages = state["messages"] + [user_msg]
            new_state_msgs = [user_msg]

    response, usage = _stream_llm(llm_messages)
    new_state_msgs.append(response)
    out["messages"] = new_state_msgs
    
    if usage:
        out["prompt_tokens"] = usage.prompt_tokens
        out["response_tokens"] = usage.completion_tokens
        out["total_tokens"] = usage.total_tokens
        
    return out

# Summarization node
def summarize_node(state: State):
    messages = state["messages"]
    
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    
    if last_human_idx == -1:
        last_human_idx = len(messages) - 1
        
    start_idx = 1 if messages and getattr(messages[0], "type", None) == "system" else 0
    msgs_to_summarize = messages[start_idx:last_human_idx]
    
    if not msgs_to_summarize:
        return {}
        
    summary_input = [SystemMessage(content=SUMMARIZATION_PROMPT)] + msgs_to_summarize
    openai_msgs = _convert_messages_to_openai(summary_input)
    response = client.chat.completions.create(
        model=orch_model,
        messages=openai_msgs
    )
    
    summary_content = response.choices[0].message.content
    removals = [RemoveMessage(id=m.id) for m in msgs_to_summarize if getattr(m, "id", None)]
    summary_msg = SystemMessage(content=f"Summary of conversation earlier:\n{summary_content}", id=str(uuid.uuid4()))
    
    return {"messages": removals + [summary_msg]}

# Determine flow based on user exit, tool calls, or continuing conversation
def should_continue(state: State) -> Literal["tool_node", "summarize_node", "orchestrator", END]:
    last_message = state["messages"][-1]
    # quit with /bye
    if last_message.content and last_message.content.strip().lower() == '/bye':
        return END
    # Route to tool_node if tool execution required
    if last_message.tool_calls:
        return "tool_node"
    if state.get("total_tokens", 0) > SUMMARIZE_AT:
        return "summarize_node"
    # Loop back to the orchestrator for user input
    return "orchestrator"

# Graph structure
workflow = StateGraph(State)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("summarize_node", summarize_node)
workflow.add_node("tool_node", tool_node)

# Set up edges
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges("orchestrator", should_continue)
workflow.add_edge("tool_node", "orchestrator")
workflow.add_edge("summarize_node", "orchestrator")

# Compile state graph application
app = workflow.compile()