from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import os

load_dotenv()

# ---------------------- 工具定义 ----------------------
@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

tools = [get_current_time, calculator, c2f]

# ---------------------- 模型 ----------------------
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
llm_with_tools = llm.bind_tools(tools)

# ---------------------- 节点 ----------------------
def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

# ---------------------- 路由 ----------------------
def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

# ---------------------- 构建图 + 检查点 + 中断配置 ----------------------
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 内存检查点
checkpointer = MemorySaver()

# ✅ 关键：interrupt_before=["tools"] 代表：即将进入tools节点前暂停
graph = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]
)

# ---------------------- 主交互（流式 + 人工审批） ----------------------
if __name__ == "__main__":
    print("LangGraph 人工审批Agent启动，输入 q 退出")
    # 同一个thread_id保证同一条对话线程，断点状态保存在这个线程
    thread_config = {"configurable": {"thread_id": "chat-001"}}

    while True:
        user_input = input("\n你：")
        if user_input.lower() == "q":
            break

        # 发给图
        for step in graph.stream(
            {"messages": [("user", user_input)]},
            config=thread_config,
            stream_mode="values"
        ):
            messages = step["messages"]
            latest_msg = messages[-1]
            print(f"\n【节点输出】类型：{latest_msg.type}")
            print(f"content：{latest_msg.content}")
            if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                print(f"👉 tool_calls：{latest_msg.tool_calls}")

        # 检查：图是否被中断（即将执行tools，等待人工确认）
        snapshot = graph.get_state(thread_config)
        if snapshot.next:
            print("\n===== ⚠️ 人工审批节点 =====")
            last_ai_msg = snapshot.values["messages"][-1]
            print("模型准备调用工具：")
            for tc in last_ai_msg.tool_calls:
                print(f"工具名称：{tc['name']}，参数：{tc['args']}")

            approve = input("是否允许执行？y 同意 / n 拒绝并说明理由：")
            if approve.lower() == "y":
                # ✅ 同意：继续执行tools节点
                print("✅ 已同意，执行工具...")
                for step in graph.stream(Command(resume=True), config=thread_config, stream_mode="values"):
                    messages = step["messages"]
                    latest_msg = messages[-1]
                    print(f"\n【节点输出】类型：{latest_msg.type}")
                    print(f"content：{latest_msg.content}")
                    if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                        print(f"👉 tool_calls：{latest_msg.tool_calls}")
            else:
                # ❌ 拒绝：把拒绝的文本作为恢复值传给图，模型会收到这条反馈
                reject_msg = approve
                print(f"❌ 拒绝执行，反馈给模型：{reject_msg}")
                for step in graph.stream(Command(resume=reject_msg), config=thread_config, stream_mode="values"):
                    messages = step["messages"]
                    latest_msg = messages[-1]
                    print(f"\n【节点输出】类型：{latest_msg.type}")
                    print(f"content：{latest_msg.content}")

        # 本轮结束，打印最终回答
        final_msg = graph.get_state(thread_config).values["messages"][-1]
        print("\n=====本轮最终回答=====")
        print(final_msg.content)
