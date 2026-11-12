from langchain_openai import ChatOpenAI
from config.settings import settings

def get_llm():
    llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model=settings.MODEL_NAME,
        temperature=0
    )
    return llm
