from loguru import logger
from config import settings
import sys
from pathlib import Path

# 自动创建日志目录
log_path = Path(settings.LOG_FILE)
log_path.parent.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)
logger.add(
    settings.LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="10MB"
)

__all__ = ["logger"]
