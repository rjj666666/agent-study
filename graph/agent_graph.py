from langgraph.graph import StateGraph, START, END
# 删掉这一行：from langgraph.types import interrupt
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from llm.zhipu_llm import llm
from tools.calculator import calculator_tool
from rag.vector_store import get_rag_search_tool


tools = [
    calculator_tool,
    get_rag_search_tool()
]

react_agent = create_react_agent(model=llm, tools=tools)

builder = StateGraph(AgentState)
builder.add_node("agent", react_agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

memory = MemorySaver()

# interrupt_before 是 compile 参数，不需要额外导入
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["agent"]
)

__all__ = ["graph"]
