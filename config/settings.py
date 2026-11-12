from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 拿到 config/settings.py 所在目录，再往上一层（项目根目录 agent_study）
BASE_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    MODEL_NAME: str 

    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "knowledge_base"

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    RETRIEVE_TOP_K: int = 3

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",   # ✅ 关键：绝对路径
        env_file_encoding="utf-8"
    )
    


settings = Settings()
