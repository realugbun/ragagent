import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", 5432)
    DB_COLLECTION_NAME = os.getenv("DB_COLLECTION_NAME", "docs")
    CHUNK_TOKEN_LIMIT = os.getenv("CHUNK_TOKEN_LIMIT", 1024)
    RAG_DIRECTORY = os.getenv("RAG_DIRECTORY", "docs")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", 6379)
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_DB = os.getenv("REDIS_DB", 0)
    REDIS_QUEUE = os.getenv("REDIS_QUEUE", "document_processing")