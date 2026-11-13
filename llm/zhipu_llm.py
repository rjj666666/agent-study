from langchain_openai import ChatOpenAI
from config import settings

llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    model=settings.MODEL_NAME,
    temperature=settings.LLM_TEMPERATURE
)

__all__ = ["llm"]
