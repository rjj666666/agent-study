
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
# 设置huggingface镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

load_dotenv()


# ========== 1. 初始化向量库（只执行一次加载、切分、入库） ==========
loader = TextLoader("./knowledge.txt", encoding="utf-8")
docs = loader.load()
# 文档切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
split_docs = text_splitter.split_documents(docs)
# 本地Embedding模型
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 向量库
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ========== 2. 定义工具：检索文档 + 计算器 + 温转换 ==========
@tool
def retrieve_doc(query: str) -> str:
    """
    当问题需要查阅本地知识库、项目介绍、学习路线时，调用这个工具检索文档。
    Args:
        query: 用户的检索关键词/问题
    """
    res_docs = retriever.invoke(query)
    content = "\n".join([d.page_content for d in res_docs])
    return f"【检索到的文档内容】:\n{content}"

@tool
def calculator(a: float, b: float, op: str) -> str:
    """
    简单计算器，支持 + - *
    Args:
        a: 数字1
        b: 数字2
        op: 运算符，只能是 + - *
    """
    if op == "+":
        res = a + b
    elif op == "-":
        res = a - b
    elif op == "*":
        res = a * b
    else:
        return "不支持该运算符，仅支持 + - *"
    return f"{a} {op} {b} = {res}"

@tool
def c2f(c: float) -> str:
    """摄氏度转华氏度"""
    f = c * 1.8 + 32
    return f"{c} ℃ = {f:.2f} ℉"

tools = [retrieve_doc, calculator, c2f]

# ========== 3. 大模型绑定工具 ==========
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
from langchain_core.callbacks import BaseCallbackHandler
import json

class PrintToolsCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        if "tools" in kwargs:
            print("\n===== 发给API完整tools =====")
            print(json.dumps(kwargs["tools"], ensure_ascii=False, indent=2))

# 后面你原来的代码不变
llm_with_tools = llm.bind_tools(tools)

# 测试调用，带上回调
resp = llm_with_tools.invoke(
    "123*456等于多少",
    config={"callbacks": [PrintToolsCallback()]}
)



# ========== 4. 图节点和路由 ==========
def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
graph = workflow.compile()

# ========== 5. 带记忆的交互循环 ==========
if __name__ == "__main__":
    print("✅ LangGraph RAG Agent 启动，输入 q 退出")
    chat_history = []
    while True:
        user_input = input("\n你：")
        if user_input.lower() == "q":
            break
        chat_history.append(("user", user_input))
        output = graph.invoke({"messages": chat_history})
        chat_history = output["messages"]
        answer = output["messages"][-1].content
        print(f"\nAgent：{answer}")
