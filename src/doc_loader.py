import os
import logging
from typing import List
import uuid

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader

from src.logger import setup_logger, log_execution_time
from src.config import Config
from src.database import get_pgvector_db
from src.embeddings import get_tokenizer, get_text_splitter, get_embedding_function, add_metadata_to_chunk, get_chunk_tags, get_text_hash

load_dotenv()

logger = setup_logger(logging.INFO)

tokenizer = get_tokenizer()
text_splitter = get_text_splitter(tokenizer=tokenizer)
embedding_function = get_embedding_function()

db = get_pgvector_db(embedding_function)

def chunk_documents(documents: List[Document]) -> List[Document]:
    docs = []
    for doc in documents:
        doc.metadata["document_id"] = str(uuid.uuid4())
        doc.metadata["document_hash"] = get_text_hash(doc)
        doc.metadata["source"] = os.path.normpath(doc.metadata["source"])
        chunks = text_splitter.split_documents([doc])
        i = 0
        for chunk in chunks:
            chunk = add_metadata_to_chunk(
                chunk=chunk, 
                doc_id=doc.metadata["document_id"], 
                document_hash=doc.metadata["document_hash"],
                source_type="file",
                index=i)
            docs.append(chunk)
            i += 1
    logger.info("Loaded documents", extra={"documents": len(documents), "chunks": len(docs)})
    return docs

def load_documents(directory: str):
    loader = DirectoryLoader(directory)
    documents = loader.load()

    chunks = chunk_documents(documents)
    logger.info("Loaded documents from directory", extra={"documents": len(documents), "directory": directory})
    return chunks



@log_execution_time(logger)
def main():

    docs = load_documents(Config.RAG_DIRECTORY)

    for doc in docs:
        result = db.similarity_search(
            query=doc.metadata["chunk_hash"],
            k=1,
            filter={"chunk_hash": {"$eq": doc.metadata["chunk_hash"]}},
        )
        if result:
            logger.debug("Chunk already in database, skipping", extra={"document_id": doc.metadata["document_id"], "chunk_hash": doc.metadata['chunk_hash']})
            continue

        doc.metadata["tags"] = get_chunk_tags(doc)

        db.add_documents([doc])

if __name__ == "__main__":
    main()
