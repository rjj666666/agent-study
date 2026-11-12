from sentence_transformers import CrossEncoder

# BGE 重排模型，轻量本地跑
rerank_model = CrossEncoder("BAAI/bge-reranker-base")

def rerank_docs(query: str, docs, top_n: int = 2):
    if not docs:
        return []
    pairs = [[query, doc.page_content] for doc in docs]
    scores = rerank_model.predict(pairs)
    # 按分数从高到低排序
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    # 过滤低分，分数太低直接丢弃
    filtered = [(d, s) for d, s in scored if s > 0.0]
    return [d for d, s in filtered[:top_n]]
