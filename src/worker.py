import io
import os

import torch
import magic

from rq import Queue, SimpleWorker
from PyPDF2 import PdfReader
from langchain_core.documents import Document

from src.config import Config
from src.database import get_db, get_redis, create_document
from src.logger import setup_logger
from src.embeddings import (
    get_tokenizer, get_text_splitter, get_text_hash, get_chunk_tags, get_tokens_in_text
)

logger = setup_logger()

# TODO: Move to embeddings.py
class EmbeddingSingleton:
    _instance = None

    def __init__(self):
        pass

    def get_instance(self):
        if EmbeddingSingleton._instance is None or os.getpid() != EmbeddingSingleton._pid:
            device = self.get_device()
            logger.info(f"Initializing embeddings on {device}")
            from langchain_huggingface import HuggingFaceEmbeddings
            EmbeddingSingleton._instance = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                encode_kwargs={'normalize_embeddings': True},
                model_kwargs={'device': device}
            )
            EmbeddingSingleton._pid = os.getpid()
        return EmbeddingSingleton._instance
    
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "cpu"
        return "cpu"

es = EmbeddingSingleton()

embedding_function = es.get_instance()  # Pre-fork load

redis_conn = get_redis()
q = Queue(Config.REDIS_QUEUE, connection=redis_conn)

def detect_file_type(file_path):
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file_path)
    if mime_type == "text/plain" or mime_type == "text/markdown":
        return "txt"  # Treat Markdown as text
    elif mime_type == "application/pdf":
        return "pdf"
    elif mime_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return "docx"
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

def extract_text(file_path, file_type):
    with open(file_path, "rb") as f:
        content = f.read()

    if file_type in ["txt", "md"]:
        return content.decode("utf-8")
    
    elif file_type == "pdf":
        reader = PdfReader(io.BytesIO(content))
        return "".join(page.extract_text() for page in reader.pages)
    
    return None

def process_document(job_id: str):
    logger.info("Processing document", extra={"job_id": job_id})
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                # Mark job as processing
                cur.execute("UPDATE jobs SET status = 'processing', start_processing_at = NOW() WHERE id = %s", (job_id,))
                conn.commit()
                
                # Fetch job details
                cur.execute("SELECT source, text, max_chunk_size, collection_id FROM jobs WHERE id = %s", (job_id,))
                job = cur.fetchone()
                if not job:
                    logger.error("Job not found", extra={"job_id": job_id})
                    raise ValueError("Job not found")
                source, text, max_chunk_size, collection_id = job
                
                # Determine source and extract text
                file_type = "text"
                source_type = "api" if text else "file"
                if source_type == "file":
                    file_type = detect_file_type(source)
                    text = extract_text(source, file_type)
                if not text:
                    logger.error("No text content available", extra={"job_id": job_id})
                    raise ValueError("No text content available")


                
                # Create LangChain Document
                initial_doc = Document(page_content=text, metadata={"source_type": source_type})
                doc_hash = get_text_hash(initial_doc)

                cur.execute("""
                    SELECT 1
                    FROM documents
                    WHERE collection_id = %s
                    AND hash = %s
                    AND is_deleted = FALSE
                    LIMIT 1
                """, (collection_id, doc_hash))
                if cur.fetchone():
                    cur.execute("UPDATE jobs SET status = 'completed' WHERE id = %s", (job_id,))
                    conn.commit()
                    logger.info("Marked job as completed due to duplicate", extra={"job_id": job_id})
                    return
                
                tokenizer = get_tokenizer()
                text_splitter = get_text_splitter(tokenizer, chunk_size=max_chunk_size)

                # Create the document object
                doc_token_count = get_tokens_in_text(text)
                doc_id = create_document(
                    hash=doc_hash,
                    collection_id=collection_id,
                    source=source,
                    file_type=file_type,
                    token_count=doc_token_count
                )
                logger.info("Created document", extra={"doc_id": doc_id, "job_id": job_id})

                chunks = text_splitter.split_documents([initial_doc])
                logger.info("Split document into chunks", extra={"chunk_count": len(chunks), "job_id": job_id})

                for index, chunk in enumerate(chunks):
                    
                    embedding = embedding_function.embed_query(chunk.page_content)
                    logger.info("Embedded chunk", extra={"chunk_index": index, "job_id": job_id})
                    token_count = get_tokens_in_text(chunk.page_content)
                    logger.info("Counted tokens in chunk", extra={"chunk_index": index, "job_id": job_id})
                    chunk_hash = get_text_hash(chunk)
                    logger.info("Hashed chunk", extra={"chunk_index": index, "job_id": job_id})
                    tags = get_chunk_tags(chunk)
                    logger.info("Extracted tags from chunk", extra={"chunk_index": index, "job_id": job_id})
                    
                    cur.execute("""
                        INSERT INTO chunks (document_id, text, tags, hash, token_count, document_index, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                    """, (doc_id, chunk.page_content, tags, chunk_hash, token_count, index, embedding))
                

                cur.execute("UPDATE jobs SET status = 'completed', document_id = %s WHERE id = %s", (doc_id, job_id))
                conn.commit()
                logger.info("Processed document ", extra={"job_id": job_id, "doc_id": doc_id})

    except Exception as e:
        logger.error("Failed to process document", extra={"job_id": job_id, "error": str(e)})
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE jobs SET status = 'failed', error_message = %s WHERE id = %s", (str(e), job_id))
                conn.commit()


if __name__ == "__main__":
    worker = SimpleWorker([q])
    worker.work()