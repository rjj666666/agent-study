from langchain_core.tools import tool
from rag.vector_store import rag_search

@tool
def rag_search_tool(query: str) -> str:
    """
    检索本地知识库，查询项目、技术相关内容，检索后会经过CrossEncoder重排
    Args:
        query: 用户检索问题
    """
    docs = rag_search(query)
    if not docs:
        return "知识库中没有找到相关内容"
    content = "\n".join([d.page_content for d in docs])
    return f"检索到内容：\n{content}"

__all__ = ["rag_search_tool"]
