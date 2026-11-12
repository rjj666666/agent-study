from typing import List, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    context: str # 用来放RAG检索出来的知识库上下文
