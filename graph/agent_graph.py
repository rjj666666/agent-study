from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from graph.state import AgentState
from llm.zhipu_llm import llm
from tools.calculator import calculator_tool
from tools.rag_search import rag_search_tool
from utils.audit_log import write_audit_log
from config.settings import AUDIT_LOG_PATH

# 工具列表
tools = [calculator_tool, rag_search_tool]
llm_with_tools = llm.bind_tools(tools)

# ========== 节点定义 ==========
def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def human_approve_node(state: AgentState):
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    print("\n===== 需要人工审批【工具调用】 =====")
    for idx, call in enumerate(tool_calls, start=1):
        print(f"【工具{idx}】名称：{call['name']}")
        print(f"【工具{idx}】参数：{call['args']}\n")
    user_input = input("输入 yes 放行执行，其他拒绝：")

    # 写入审计日志
    approve_result = "同意" if user_input.strip().lower() == "yes" else "拒绝"
    write_audit_log(AUDIT_LOG_PATH, tool_calls, approve_result)

    if user_input.strip().lower() == "yes":
        return {"messages": [last_msg]}
    else:
        refuse_msg = HumanMessage(content="人工拒绝本次工具调用，停止执行。")
        return {"messages": [refuse_msg]}

# ========== 路由 ==========
def route_after_agent(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "human_approve"
    return END

def route_after_approve(state: AgentState):
    last_msg = state["messages"][-1]
    if "人工拒绝本次工具调用" in last_msg.content:
        return END
    return "tools"

# ========== 编译图 ==========
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("human_approve", human_approve_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent, {"human_approve": "human_approve", END: END})
workflow.add_conditional_edges("human_approve", route_after_approve, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

graph = workflow.compile()
__all__ = ["graph"]
