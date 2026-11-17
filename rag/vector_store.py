from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from pathlib import Path
from config.settings import EMBED_MODEL_NAME
from .loader import load_txt_file, load_pdf_file
from .rerank import rerank_docs

# 全局向量库实例
vector_db = None

def get_embedding():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def init_knowledge(knowledge_file: Path, persist_dir: Path):
    """初始化知识库，加载txt/pdf，构建向量库，赋值给全局vector_db"""
    global vector_db
    file_path_str = str(knowledge_file)
    if file_path_str.lower().endswith(".txt"):
        docs = load_txt_file(file_path_str)
    elif file_path_str.lower().endswith(".pdf"):
        docs = load_pdf_file(file_path_str)
    else:
        raise ValueError("仅支持 txt / pdf 文件")

    embeddings = get_embedding()
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir)
    )
    return vector_db

def rag_search(query: str, top_k: int = 5):
    """rag检索 + CrossEncoder重排（复用你写好的rerank_docs）"""
    if vector_db is None:
        raise RuntimeError("请先调用 init_knowledge 初始化知识库")
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    raw_docs = retriever.invoke(query)
    # 这里直接传给你原来的rerank，自动读取settings里的RERANK_TOP_N
    reranked_docs = rerank_docs(query, raw_docs)
    return reranked_docs

