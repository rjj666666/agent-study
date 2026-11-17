from langchain_core.tools import tool
from asteval import Interpreter

@tool
def calculator(expression: str) -> str:
    """
    数学计算器，可以计算表达式，例如：2*3.14*5、(100+20)/5
    只支持基础数学运算，禁止执行系统命令
    Args:
        expression: 数学表达式字符串
    """
    try:
        aeval = Interpreter()
        res = aeval(expression)
        return f"计算结果：{res}"
    except Exception as e:
        return f"计算失败：{str(e)}"

calculator_tool = calculator
__all__ = ["calculator_tool"]
