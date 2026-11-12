from loguru import logger
import sys

# 日志配置
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add("agent.log", rotation="10 MB", encoding="utf-8")

if __name__ == "__main__":
    logger.info("日志测试")
