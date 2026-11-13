from sentence_transformers import CrossEncoder
from config import settings
from utils.logger import logger

rerank_model = CrossEncoder(settings.RERANK_MODEL_NAME)

def rerank_docs(query: str, docs, top_n: int = None):
    top_n = top_n or settings.RERANK_TOP_N
    pairs = [[query, doc.page_content] for doc in docs]
    scores = rerank_model.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    selected = [item[0] for item in scored_docs[:top_n]]
    logger.info(f"重排后选取 {len(selected)} 条文档")
    return selected
