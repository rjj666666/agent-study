# 放在文件最顶部，所有import之前！
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings
from rag.loader import load_and_split_docs
from utils.logger import logger

# 本地嵌入模型，不走智谱的embedding接口
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vector_store() -> Chroma:
    db = Chroma(

        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embedding_model,
    )
    return db

def init_knowledge_base(file_path: str = "knowledge.txt"):
    db = get_vector_store()
    docs = load_and_split_docs(file_path, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if docs:
        db.add_documents(docs)
        logger.info(f"入库 {len(docs)} 个分片")
    return db

if __name__ == "__main__":
    init_knowledge_base("knowledge.txt")
