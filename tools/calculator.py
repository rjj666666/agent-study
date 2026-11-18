from langchain_core.tools import tool
import re

# 白名单：只允许数字、小数点、加减乘除、括号、空格
SAFE_PATTERN = re.compile(r"^[\d\.\+\-\*\/\(\)\s]+$")

@tool
def calculator(expression: str) -> str:
    """
    数学计算器，可以计算表达式，例如：2*3.14*5、(100+20)/5
    仅支持 + - * / 和括号，不支持函数、导入、属性访问
    Args:
        expression: 数学表达式字符串
    """
    expr = expression.strip()
    if not SAFE_PATTERN.match(expr):
        return "计算失败：表达式包含非法字符/危险操作"
    try:
        res = eval(expr)
        return f"计算结果：{res}"
    except Exception as e:
        return f"计算失败：{str(e)}"

calculator_tool = calculator
__all__ = ["calculator_tool"]
