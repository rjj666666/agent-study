from datetime import datetime
from pathlib import Path

def write_audit_log(log_path: Path, tool_calls: list, approve_result: str):
    """
    写入工具审批审计日志
    :param log_path: 日志文件路径（Path对象）
    :param tool_calls: 工具调用信息
    :param approve_result: "同意"/"拒绝"
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{now}] tool_calls={tool_calls} | approve={approve_result}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line)
