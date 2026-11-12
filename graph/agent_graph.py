from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from config.settings import settings
from rag.vector_store import get_vector_store
from tools.calculator import calculator
from graph.state import AgentState
from utils.logger import logger
from rag.rerank import rerank_docs
from llm.zhipu_llm import get_llm

import json
import re

# ===================== 工具解析辅助函数 =====================
def extract_json(text: str):
    """提取模型输出里的JSON，清洗多余文字"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group()
    return text

def is_tool_call(msg):
    content = msg.content.strip()
    json_str = extract_json(content)
    try:
        data = json.loads(json_str)
        if data.get("name") == "calculator":
            return True, data
    except Exception:
        pass
    return False, None

# 自定义路由，替代内置 tools_condition
def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    flag, _ = is_tool_call(last_msg)
    if flag:
        return "tools"
    return END


# 把所有工具放到列表（这里只是存着，不再bind_tools）
tools = [calculator]
llm = get_llm()
llm_with_tools = llm

# ===================== 节点定义 =====================
def retrieve_node(state: AgentState):
    """检索 + BGE重排节点"""
    last_msg = state["messages"][-1]
    query = last_msg.content
    vector_store = get_vector_store()
    raw_docs = vector_store.similarity_search(query, k=4)
    # BGE重排
    reranked_docs = rerank_docs(query, raw_docs)
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    return {"context": context}

def agent_node(state: AgentState):
    context = state.get("context", "")
    system_prompt = """
你是RAG问答助手，可以使用计算器工具做数学运算。
知识库内容：{context}

当需要计算时，严格只输出如下JSON，不要额外文字、不要解释、不要markdown：
{{"name": "calculator", "parameters": {{"a": 数字, "b": 数字, "op": "+-*/其中一个"}}}}
不需要计算、直接回答问题时，正常用自然语言回答。
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])
    chain = prompt | llm_with_tools
    resp = chain.invoke({
        "messages": state["messages"],
        "context": context
    })
    return {"messages": [resp]}

def tool_node(state: AgentState):
    """执行计算器"""
    last_msg = state["messages"][-1]
    _, tool_data = is_tool_call(last_msg)
    a = tool_data["parameters"]["a"]
    b = tool_data["parameters"]["b"]
    op = tool_data["parameters"]["op"]
    result = calculator(a, b, op)
    return {"messages": [ToolMessage(content=str(result), tool_call_id="fake_001")]}

def need_retrieve(state: AgentState):
    """判断是否需要检索知识库，你可以根据自己提示词调整，这里默认：走检索"""
    # 简单策略：全部先检索；如果你想做路由判断，可以在这里改逻辑
    return "retrieve"

# ===================== 构建图 =====================
def build_graph():
    builder = StateGraph(AgentState)
    # 注册节点
    builder.add_node("agent", agent_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("tools", tool_node)

    # 入口
    builder.set_entry_point("agent")

    # agent -> 判断是否检索
    builder.add_conditional_edges(
        "agent",
        need_retrieve,
        {
            "retrieve": "retrieve",
            "agent": "agent"
        }
    )
    builder.add_edge("retrieve", "agent")

    # agent -> 判断是否调用工具
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    builder.add_edge("tools", "agent")

    graph = builder.compile()
    return graph

graph = build_graph()
