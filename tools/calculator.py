from langchain_core.tools import tool

# tools/calculator.py
def calculator(a: float, b: float, op: str) -> float:
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            return "除数不能为0"
        return a / b
    else:
        return "不支持的运算符"
