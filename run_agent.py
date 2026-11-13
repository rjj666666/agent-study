from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from typing import Annotated, Sequence
from typing_extensions import TypedDict
import operator
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import datetime

# 加载 .env 文件
load_dotenv()

# ===================== 配置区（从你的.env读取） =====================
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "glm-5.3-flash")

EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHROMA_PERSIST_DIR = "./chroma_db"
KNOWLEDGE_FILE = "./knowledge.txt"
LOG_FILE = "./audit.log"

# ===================== LLM 初始化（智谱兼容openai格式） =====================
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0
)

# ===================== 向量库 & RAG 初始化 =====================
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def build_vector_db():
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        doc_text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(doc_text)
    print(f"INFO | txt文件 {KNOWLEDGE_FILE} 切分得到 {len(chunks)} 块")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("INFO | 知识库写入chroma完成（新版Chroma自动持久化）")
    return vectorstore

vector_db = build_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# ===================== 定义工具 =====================
@tool
def calculator(a: float, b: float) -> str:
    """
    用于两个数字相乘计算
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    res = a * b
    return f"{a} × {b} = **{res}**"

@tool
def knowledge_retrieve(query: str) -> str:
    """
    检索本地知识库knowledge.txt，查询Agent开发、技术栈相关内容
    Args:
        query: 用户检索问题
    """
    docs = retriever.invoke(query)
    if not docs:
        return "知识库中没有找到相关内容"
    content = "\n".join([d.page_content for d in docs])
    return f"检索到内容：\n{content}"

tools = [calculator, knowledge_retrieve]
llm_with_tools = llm.bind_tools(tools)

# ===================== 状态定义 =====================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ===================== Graph 节点 =====================
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
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    approve_result = "同意" if user_input.strip().lower() == "yes" else "拒绝"
    log_line = f"[{now}] tool_calls={tool_calls} | approve={approve_result}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

    if user_input.strip().lower() == "yes":
        return {"messages": [last_msg]}
    else:
        refuse_msg = HumanMessage(content="人工拒绝本次工具调用，停止执行。")
        return {"messages": [refuse_msg]}

# ===================== 路由逻辑 =====================
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

# ===================== 编译图 =====================
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("human_approve", human_approve_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent, {"human_approve": "human_approve", END: END})
workflow.add_conditional_edges("human_approve", route_after_approve, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# ===================== 主循环 =====================
if __name__ == "__main__":
    print("✅ Agent启动成功，输入 exit 退出")
    config = {"configurable": {"thread_id": "1"}}
    while True:
        query = input("\n请输入你的问题：")
        if query.strip().lower() in ["exit", "quit"]:
            print("程序退出")
            break
        inputs = {"messages": [HumanMessage(content=query)]}
        final_msg = None
        for chunk in graph.stream(inputs, config=config):
            for node_name, value in chunk.items():
                if "messages" in value:
                    final_msg = value["messages"][-1]
        print("\n【最终回答】")
        print(final_msg.content)
