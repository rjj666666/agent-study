from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from config import settings
from utils.logger import logger

def load_txt_file(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    logger.info(f"txt文件 {file_path} 切分得到 {len(chunks)} 块")
    return chunks

def load_pdf_file(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    logger.info(f"pdf文件 {file_path} 切分得到 {len(chunks)} 块")
    return chunks
