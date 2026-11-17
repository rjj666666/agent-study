from pathlib import Path
import os
from dotenv import load_dotenv

LLM_TEMPERATURE = 0

# 项目根目录（当前文件向上一层，也就是 agent_study/）
BASE_DIR = Path(__file__).parent.parent

# 加载根目录下的 .env
load_dotenv(BASE_DIR / ".env")

# ========== 智谱 LLM 配置 ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "glm-5.3-flash")

# ========== RAG & 向量库配置 ==========
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
KNOWLEDGE_FILE = BASE_DIR / "data" / "knowledge.txt"

# ========== 审计日志配置 ==========
AUDIT_LOG_PATH = BASE_DIR / "logs" / "audit.log"

# 日志配置
LOG_FILE = BASE_DIR / "logs" / "agent.log"

# 重排配置（之前也要的，一起放这里）
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
RERANK_TOP_N = 3