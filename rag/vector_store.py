from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from config import settings
from rag.loader import load_txt_file
from rag.rerank import rerank_docs
from utils.logger import logger

embedding = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

vector_db = Chroma(
    persist_directory=settings.CHROMA_PERSIST_PATH,
    collection_name=settings.CHROMA_COLLECTION_NAME,
    embedding_function=embedding
)

def init_knowledge(file_path: str = "./knowledge.txt"):
    chunks = load_txt_file(file_path)
    vector_db.add_documents(chunks)
    logger.info("知识库写入chroma完成")

@tool
def rag_search(query: str) -> str:
    """
    检索内部知识库，用于回答和知识库相关的问题
    Args:
        query: 用户检索问题
    """
    raw_docs = vector_db.similarity_search(query, k=settings.RAG_TOP_K)
    if not raw_docs:
        return "知识库没有找到相关内容"
    reranked_docs = rerank_docs(query, raw_docs)
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    return context

def get_rag_search_tool():
    return rag_search

__all__ = ["vector_db", "init_knowledge", "rag_search", "get_rag_search_tool"]
