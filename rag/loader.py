from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger

def load_and_split_docs(file_path: str, chunk_size: int, chunk_overlap: int):
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        logger.info(f"成功加载文档：{file_path}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(docs)
        logger.info(f"文档切片完成，一共 {len(split_docs)} 块")
        return split_docs
    except Exception as e:
        logger.error(f"文档加载失败: {str(e)}")
        raise e

if __name__ == "__main__":
    from config.settings import settings
    res = load_and_split_docs("knowledge.txt", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    print(res[0])
