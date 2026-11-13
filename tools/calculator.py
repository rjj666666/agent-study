from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    数学计算器，可以计算表达式，例如：2*3.14*5、(100+20)/5
    Args:
        expression: 数学表达式字符串
    """
    try:
        res = eval(expression)
        return f"计算结果：{res}"
    except Exception as e:
        return f"计算失败：{str(e)}"

calculator_tool = calculator
__all__ = ["calculator_tool"]
