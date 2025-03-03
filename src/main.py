import os
from typing import List
import uuid

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, status, UploadFile
from pydantic import BaseModel
from rq import Queue
import shutil
from langchain_core.documents import Document

from src.config import Config
import src.database as db
from src.embeddings import get_embedding_function, get_text_splitter, get_tokenizer, add_metadata_to_chunk, get_chunk_tags, get_text_hash
from src.logger import setup_logger

logger = setup_logger()

app = FastAPI()
embedding_function = get_embedding_function()
tokenizer = get_tokenizer()

redis_conn = db.get_redis()
q = Queue(Config.REDIS_QUEUE, connection=redis_conn)

def is_valid_uuid(uuid_string:str, version:int =4) -> bool:
    """
    Checks if a string is a valid UUID of a specific version.

    Args:
        uuid_string: The string to check.
        version: The UUID version to validate against (default is 4).

    Returns:
        True if the string is a valid UUID of the specified version, False otherwise.
    """
    try:
        uuid_obj = uuid.UUID(uuid_string, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_string

def format_document_response(results, correlation_id: str = None):
    return {
        "metadata": {
            "correlation_id": correlation_id,
            "token_count": sum([int(result[8]) for result in results]),
            "chunk_count": len(results)
        },
        "data": [
            {
                "chunk_id": result[0],
                "chunk_hash": result[1],
                "document_id": result[2],
                "document_hash": result[3],
                "tags": result[4] if result[4] is not None else [],
                "source": result[5],
                "document_index": int(result[6]),
                "file_type": result[7],
                "token_count": int(result[8]),
                "created_at": result[9],
                "updated_at": result[10],
                "text": result[11]
            }
            for result in results
        ]
    }

#################################################
#                  SEARCH                       #
#################################################

class DocumentRequest(BaseModel):
    query: str
    max_results: int = 5


@app.post("/v1/search")
def search_vector_store(request: Request, document_request: DocumentRequest):
    correlation_id = getattr(request.state, "correlation_id", None)
    query = document_request.query.strip()
    max_results = document_request.max_results

    if not query:
        logger.error("Query cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")
    

    try:
        # Generate the query embedding
        query_embedding = embedding_function.embed_query(query)

        # Search the vector store
        results = db.similarity_search(query_embedding, max_results)                
        response = {
            "metadata": {
                "correlation_id": correlation_id,
                "token_count": sum([int(result[8]) for result in results]),
                "chunk_count": len(results)
            },
            "data": [
                {
                    "chunk_id": result[0],
                    "chunk_hash": result[1],
                    "document_id": result[2],
                    "document_hash": result[3],
                    "tags": result[4] if result[4] is not None else [],
                    "source": result[5],
                    "document_index": int(result[6]),
                    "file_type": result[7],
                    "token_count": int(result[8]),
                    "created_at": result[9],
                    "updated_at": result[10],
                    "text": result[11],
                    "similarity_score": result[12]
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error("Error searching vector store", extra={"correlation_id":correlation_id,"error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

    return response

class DocumentRequestByTag(BaseModel):
    tags: List[str]
    max_results: int = 5

@app.post("/v1/search/tags")
def search_by_tag(request: Request, document_request: DocumentRequestByTag):
    correlation_id = getattr(request.state, "correlation_id", None)
    tags = document_request.tags
    max_results = document_request.max_results

    if not tags:
        logger.error("Query cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")
    
    tags = [tag.strip() for tag in tags if tag.strip()]

    if not tags:
        logger.error("Query contains only empty strings after trimming", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query contains only empty strings after trimming")
    
    try:
        results = db.search_by_tags(tags, max_results)
        if not results:
            logger.error("Tags not found", extra={"correlation_id": correlation_id, "document_hash": tags})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tags not found")
        
        response = format_document_response(results, correlation_id)
    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        logger.error("Error searching vector store", extra={"correlation_id":correlation_id,"error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

    return response

#################################################
#                  DOCUMENT                     #
#################################################

class CreateDocumentRequest(BaseModel):
    document: str
    token_limit: int = int(Config.CHUNK_TOKEN_LIMIT)

# @app.post("/v1/document", status_code=status.HTTP_201_CREATED)
# def create_document(request: Request, document_request: CreateDocumentRequest):
#     correlation_id = getattr(request.state, "correlation_id", None)
#     content = document_request.document
#     token_limit = document_request.token_limit

#     if not content:
#         logger.error("Content cannot be empty", extra={"correlation_id": correlation_id})
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content cannot be empty")
    
#     doc = Document(content)
#     doc_hash = get_text_hash(doc)

    
#     is_exists = db.search_by_document_hash(get_text_hash(Document(content)))
#     if is_exists:
#         logger.error("Document already exists", extra={"correlation_id": correlation_id, "document_id": is_exists[0][2]})
#         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document already exists with id {}".format(is_exists[0][2]))
    
#     text_splitter = get_text_splitter(tokenizer, chunk_size=token_limit)
    
#     chunks = text_splitter.split_documents([doc])
#     i = 0

#     try:
#         doc_id = str(uuid.uuid4())
#         for chunk in chunks:
#             chunk = add_metadata_to_chunk(
#                 chunk=chunk, 
#                 doc_id=doc_id,
#                 document_hash=doc_hash,
#                 source_type="api",
#                 index=i,
#             )
#             chunk.metadata["tags"] = get_chunk_tags(chunk)
#             i += 1
        
#         vectordb.add_documents(chunks)

#         results = db.search_by_document_hash(doc_hash)
#         response = format_document_response(results, correlation_id)

#     except Exception as e:
#         logger.error("Error creating chunks from document", extra={"correlation_id": correlation_id, "error": str(e)})
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    
#     return response


@app.post("/v1/document/async", status_code=status.HTTP_202_ACCEPTED)
async def create_document_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),           # Optional file upload
    text: str = Form(None),                  # Optional text as form data
    max_chunk_size: int = Form(1024),           # Optional max_chunk_size as form data
):
    correlation_id = getattr(request.state, "correlation_id", None)
    
    # Validate input: file or text, not both, not neither
    # We might want to think about setting a min token size to keep from overloading the system
    if file and text:
        logger.error("Cannot provide both file and text", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot provide both file and text")
    if not file and not text:
        logger.error("Must provide file or text", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must provide file or text")
    if max_chunk_size < Config.MIN_CHUNK_SIZE:
        logger.error("Max chunk size too small", extra={"correlation_id": correlation_id, "max_chunk_size": max_chunk_size, "system_min": Config.MIN_CHUNK_SIZE})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"max_chunk_size must be at least {Config.MIN_CHUNK_SIZE}")
    
    # The collection id will eventually be tied to users so each has their own collection
    collection_id = db.get_collection_id(Config.DB_COLLECTION_NAME)
    if not collection_id:
        logger.error("Collection not found", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Collection not found")
    
    if file:
        if file.content_type not in ["application/pdf", "text/plain"]:
            logger.error("Unsupported file type", extra={"correlation_id": correlation_id})
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")
        file_path = f"./uploads/{str(uuid.uuid4())}_{file.filename}"
        os.makedirs("./uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        job_id = db.create_job(max_chunk_size=max_chunk_size, collection_id=collection_id, source=file_path)
    else:
        job_id = db.create_job(max_chunk_size=max_chunk_size, collection_id=collection_id, source="text", text=text)
    
    background_tasks.add_task(q.enqueue, "src.worker.process_document", job_id)
    logger.info(f"Enqueued job {job_id}", extra={"correlation_id": correlation_id})
    
    return {
        "metadata": {"correlation_id": correlation_id},
        "data": {"job_id": job_id}
    }

@app.get("/v1/document/hash/{document_hash}")
def get_document_by_hash(request: Request, document_hash: str):
    correlation_id = getattr(request.state, "correlation_id", None)
    if not document_hash:
        logger.error("Document hash cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document hash cannot be empty")
    
    try: 
        results = db.search_by_document_hash(document_hash)
        if not results:
            logger.error("Document not found", extra={"correlation_id": correlation_id, "document_hash": document_hash})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        
        response = format_document_response(results, correlation_id)

    except Exception as e:
        logger.error("Error fetching document by hash", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

    return response

@app.get("/v1/document/{document_id}")
def get_document_by_id(request: Request, document_id: str):
    correlation_id = getattr(request.state, "correlation_id", None)
    if not document_id:
        logger.error("Document hash cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document hash cannot be empty")
    
    try: 
        results = db.search_by_document_id(document_id)
        if not results:
            logger.error("Document not found", extra={"correlation_id": correlation_id, "document_id": document_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        
        response = format_document_response(results, correlation_id)

    # Prevent the 404 from returning a 500
    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        logger.error("Error fetching document by id", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

    return response

@app.delete("/v1/document/{document_id}")
def delete_document_by_id(request: Request, document_id: str):
    correlation_id = getattr(request.state, "correlation_id", None)
    if not document_id:
        logger.error("Document hash cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document hash cannot be empty")
    
    try:
        results = db.search_by_document_id(document_id)
        if not results:
            logger.error("Document not found", extra={"correlation_id": correlation_id, "document_id": document_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        
        db.delete_by_document_id(document_id)
        response = format_document_response(results, correlation_id)

    # Prevent the 404 from returning a 500
    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        logger.error("Error deleting document by id", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    
    return response

class PreviewDocumentRequest(BaseModel):
    max_chunk_size: int = 1024
    text: str
    tags: bool = False
    token_count: bool = False

@app.post("/v1/document/preview")
def preview_document(request: Request, document_request: PreviewDocumentRequest):
    correlation_id = getattr(request.state, "correlation_id", None)
    content = document_request.text
    max_chunk_size = document_request.max_chunk_size
    is_token_count = document_request.token_count
    is_tags = document_request.tags

    if not content:
        logger.error("Content cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content cannot be empty")
    
    if max_chunk_size < Config.MIN_CHUNK_SIZE:
        logger.error("Max chunk size too small", extra={"correlation_id": correlation_id, "max_chunk_size": max_chunk_size, "system_min": Config.MIN_CHUNK_SIZE})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"max_chunk_size must be at least {Config.MIN_CHUNK_SIZE}")
    
    doc = Document(content)
    text_splitter = get_text_splitter(tokenizer, chunk_size=max_chunk_size)
    doc_hash = get_text_hash(doc)
    chunks = text_splitter.split_documents([doc])
    i = 0
    for chunk in chunks:
        chunk = add_metadata_to_chunk(
            chunk=chunk, 
            doc_id="preview",
            document_hash=doc_hash,
            source_type="preview",
            index=i,
            token_count=is_token_count,
        )
        if is_tags:
            chunk.metadata["tags"] = get_chunk_tags(chunk)
        i += 1
    response = {
        "metadata": {
            "correlation_id": correlation_id,
            **({"token_count": sum([chunk.metadata["token_count"] for chunk in chunks])} if is_token_count else {}),
            "chunk_count": len(chunks)
        },
        "data": [
            {
                "chunk_hash": chunk.metadata["chunk_hash"],
                "document_hash": chunk.metadata["document_hash"],
                **({"tags": chunk.metadata["tags"]} if is_tags else {}),
                "source": "preview",
                "document_index": chunk.metadata["chunk_index"],
                "file_type": "text",
                **({"token_count": chunk.metadata["token_count"]} if is_token_count else {}),
                "created_at": chunk.metadata["created_at"],
                "updated_at": chunk.metadata["updated_at"],
                "text": chunk.page_content,

            }
            for chunk in chunks
        ]
    }
    return response

#################################################
#                  CHUNK                        #
#################################################

@app.get("/v1/chunk/hash/{chunk_hash}")
def get_chunk_by_hash(request: Request, chunk_hash: str):
    correlation_id = getattr(request.state, "correlation_id", None)
    if not chunk_hash:
        logger.error("Chunk hash cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chunk hash cannot be empty")
    
    try:
        results = db.get_chunk_by_hash(chunk_hash)
        if not results:
            logger.error("Chunk not found", extra={"correlation_id": correlation_id, "chunk_hash": chunk_hash})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
        
        response = format_document_response(results, correlation_id)

    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        logger.error("Error fetching chunk by hash", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    
    return response

@app.get("/v1/chunk/{chunk_id}")
def get_chunk_by_id(request: Request, chunk_id: str):
    correlation_id = getattr(request.state, "correlation_id", None)
    if not chunk_id:
        logger.error("Chunk id cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chunk id cannot be empty")
    
    try:
        results = db.get_chunk_by_id(chunk_id)
        if not results:
            logger.error("Chunk not found", extra={"correlation_id": correlation_id, "chunk_id": chunk_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
        
        response = format_document_response(results, correlation_id)

    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        logger.error("Error fetching chunk by id", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    
    return response

class UpdateTagsRequest(BaseModel):
    tags: List[str]  # List of new tags to be updated

@app.put("/v1/chunk/tags/{chunk_id}")
async def update_document_tags(chunk_id: str, tags_request: UpdateTagsRequest, request: Request):
    correlation_id = getattr(request.state, "correlation_id", None)  # Generate correlation ID if missing
    new_tags = tags_request.tags
    
    if not new_tags:
        logger.error("Tags cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tags cannot be empty")
    
    new_tags = [tag.strip() for tag in new_tags if tag.strip()]

    if not new_tags:
        logger.error("Query contains only empty strings after trimming", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query contains only empty strings after trimming")

    try:
        is_exists = db.get_chunk_by_id(chunk_id)
        if not is_exists:
            logger.error("Chunk not found", extra={"correlation_id": correlation_id, "chunk_id": chunk_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
        
        # Call the database function to update the tags
        updated_row = db.update_chunk_tags(chunk_id, new_tags)
        
        # Prepare the response
        response = format_document_response([updated_row], correlation_id)
    except ValueError as e:
        logger.error(f"Invalid tags input: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tags input")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Error updating document tags", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

    return response


@app.delete("/v1/chunk/{chunk_id}")
def delete_chunk_by_id(request: Request, chunk_id: str):
    correlation_id = getattr(request.state, "correlation_id", None)
    if not chunk_id:
        logger.error("Chunk id cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chunk id cannot be empty")
    
    try:
        results = db.get_chunk_by_id(chunk_id)
        if not results:
            logger.error("Chunk not found", extra={"correlation_id": correlation_id, "chunk_id": chunk_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
        
        db.delete_chunk_by_id(chunk_id)
        response = format_document_response(results, correlation_id)

    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        logger.error("Error deleting chunk by id", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    
    return response

#################################################
#                  JOB                          #
#################################################

@app.get("/v1/job/{job_id}")
async def get_job_status(request: Request, job_id: str):
    correlation_id = getattr(request.state, "correlation_id", None)

    if not is_valid_uuid(job_id):
        logger.error("Invalid job ID", extra={"correlation_id": correlation_id, "job_id": job_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job ID")
    
    job = db.get_job_status(job_id)
    if not job:
        logger.error("Job not found", extra={"correlation_id": correlation_id, "job_id": job_id})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    
    job_id, job_status, start_processing_at, document_id, error_message, created_at, updated_at = job
    job_data = {
        "job_id": job_id,
        "created_at": created_at.isoformat(),
        "status": job_status,
    }

    if status == "failed":
        job_data["error_message"] = error_message

    if status == "completed" and updated_at:
        processing_time = (updated_at - start_processing_at).total_seconds()
        job_data["processing_time"] = processing_time  # Seconds
        job_data["document_id"] = document_id
        
    return {
        "metadata": {"correlation_id": correlation_id},
        "data": job_data
    }

#################################################
#                  HEALTH                       #
#################################################

@app.get("/v1/health")
def health_check():
    redis_status = db.ping_redis()
    db_status = db.ping_db()

    health = {
        "status": "ok" if redis_status and db_status else "error",
        "redis": "ok" if redis_status else "error",
        "database": "ok" if db_status else "error",
    }
    
    if not (redis_status and db_status):
        logger.error("Health check failed", extra={"redis": redis_status, "database": db_status})
        raise HTTPException(status_code=503, detail=health)
    
    return health

#################################################
#                  MIDDLEWARE                   #
#################################################

@app.middleware("http")
async def log_request_data(request: Request, call_next):
    # Get the correlation ID from request.state (set by the previous middleware)
    correlation_id = getattr(request.state, "correlation_id", None)
    
    # Log the request data including the correlation ID
    logger.info("Request received", extra={"method": request.method, "path": request.url.path, "correlation_id": correlation_id})
    
    # Process the request
    response = await call_next(request)
    
    # Log the response data (e.g., status code)
    logger.info(f"Response sent with status {response.status_code}", extra={"correlation_id": correlation_id})
    
    return response


@app.middleware("http")
async def add_correlation_id_header(request: Request, call_next):
    correlation_id = request.headers.get('X-Correlation-Id', str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers.raw.append((b'X-Correlation-Id', correlation_id.encode('utf-8')))
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)