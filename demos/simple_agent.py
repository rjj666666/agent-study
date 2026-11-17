from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
import datetime
# 加载密钥
load_dotenv()

# 定义工具
@tool
def calculator(a: float, b: float, op: str) -> str:
    """
    计算器工具，用来做加减乘除数学计算
    Args:
        a: 第一个数字
        b: 第二个数字
        op: 运算符，支持 + - * /
    """
    if op == "+":
        res = a + b
    elif op == "-":
        res = a - b
    elif op == "*":
        res = a * b
    elif op == "/":
        if b == 0:
            return "除数不能为0"
        res = a / b
    else:
        return "不支持该运算符"
    return f"计算结果：{a} {op} {b} = {res}"

@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def c2f(c: float) -> str:
    """
    摄氏度转华氏度
    Args:
        c: 摄氏温度
    """
    f = c * 1.8 + 32
    return f"{c} ℃ = {f:.2f} ℉"

@tool
def get_weekday(date_str:str="") -> str:
    """
    查询某一天是星期几，不传参数默认查今天
    Args:
        date_str: 日期，格式 YYYY-MM-DD
    """
    if not date_str:
        today = datetime.date.today()
    else:
        today = datetime.datetime.strptime(date_str,"%Y-%m-%d").date()
    wd = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
    return f"{today} 是 {wd[today.weekday()]}"

tools = [calculator, get_current_time, c2f, get_weekday]

# 初始化智谱GLM模型
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0
)

# 创建React Agent（新版官方推荐，底层就是LangGraph）
agent = create_agent(llm, tools)


if __name__ == "__main__":
    question = "2026-11-11是周几, 现在几点，22摄氏度等于多少华氏度，再算 15+78"
    response = agent.invoke({"messages": [("user", question)]})

    # 打印完整全过程（包含工具调用信息）
    for idx, msg in enumerate(response["messages"]):
        print(f"\n---------- 第 {idx} 条消息 ----------")
        print(f"类型：【{msg.type}】")
        print(f"content：{msg.content}")
        # 如果是AI消息，打印工具调用
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"👉 工具调用详情：{msg.tool_calls}")

    print("\n=====最终回答=====")
    print(response["messages"][-1].content)
