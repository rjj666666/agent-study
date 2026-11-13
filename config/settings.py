from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # 智谱（OpenAI兼容调用方式）
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "glm-5.3-flash")
    LLM_TEMPERATURE: float = 0.0

    # Chroma向量库
    CHROMA_PERSIST_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "knowledge_base"

    # Embedding & Rerank
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"
    RERANK_MODEL_NAME: str = "BAAI/bge-reranker-base"
    RAG_TOP_K: int = 5
    RERANK_TOP_N: int = 3

    # 日志
    LOG_FILE: str = "./agent.log"

settings = Settings()
