import yaml
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
orch_model = config.get("default_orch_model")
llm = ChatOllama(model=orch_model, temperature=0)

tools = []
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def orchestrator_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(State)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges("orchestrator", should_continue)
workflow.add_edge("tool_node", "orchestrator")

app = workflow.compile()

if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="What is the weather in San Francisco?")]}
    for chunk in app.stream(inputs, stream_mode="values"):
        for message in chunk.get("messages", []):
            message.pretty_print()