# Standard library imports
from datetime import datetime
import os
import shutil
import uuid
from itertools import chain
from typing import List, Optional, TypeVar, Generic

# Third-party imports
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, status, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from rq import Queue

# Local imports
from src.config import Config
import src.database as db
from src.embeddings import (
    get_embedding_function,
    get_text_splitter,
    get_tokenizer,
    add_metadata_to_chunk,
    get_chunk_tags,
    get_text_hash
)
from src.logger import setup_logger


logger = setup_logger()
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

# ---------------------------
# Metadata Models
# ---------------------------
class BaseMetadata(BaseModel):
    """Base metadata with just the correlation ID."""
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking the request")

class FullMetadata(BaseMetadata):
    """Extended metadata that includes token count, chunk count, and related tags."""
    token_count: int = Field(..., description="Total token count from the results")
    chunk_count: int = Field(..., description="Number of document chunks returned")
    related_tags: List[str] = Field(default_factory=list, description="Aggregated related tags from the results")

# ---------------------------
# Error Models
# ---------------------------
class HTTPError(BaseModel):
    """Model for error details."""
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")

class HTTPErrorResponse(BaseModel):
    """Response model for errors."""
    metadata: BaseMetadata  # Or your own minimal metadata model
    error: HTTPError

# ---------------------------
# Health Check Models
# ---------------------------
class HealthData(BaseModel):
    status: str = Field(..., description="Overall health status")
    redis: str = Field(..., description="Redis status")
    database: str = Field(..., description="Database status")

class HealthResponse(BaseModel):
    metadata: BaseMetadata
    data: HealthData

# ---------------------------
# Request Models
# ---------------------------
class DocumentRequest(BaseModel):
    """Request model for search by query."""
    query: str = Field(..., description="The search query string", example="Find AI papers")
    max_results: int = Field(5, description="Maximum number of results to return", example=5)

class DocumentRequestByTag(BaseModel):
    """Request model for search by tags."""
    tags: List[str] = Field(..., description="A list of tags to search for", example=["machine learning", "NLP"])
    max_results: int = Field(5, description="Maximum number of results to return", example=5)

class PreviewDocumentRequest(BaseModel):
    """Request model for previewing a document."""
    max_chunk_size: int = Field(1024, description="Maximum chunk size", example=1024)
    text: str = Field(..., description="Document text to preview", example="Some text to preview")
    tags: bool = Field(False, description="Whether to compute tags", example=False)
    token_count: bool = Field(False, description="Whether to include token counts", example=False)

class UpdateTagsRequest(BaseModel):
    """Request model for updating chunk tags."""
    tags: List[str] = Field(..., description="List of new tags for the chunk", example=["newtag1", "newtag2"])

# ---------------------------
# Response Models
# ---------------------------
class BaseSearchResult(BaseModel):
    """Common fields for search results."""
    chunk_id: uuid.UUID = Field(..., description="Unique identifier for the chunk (UUIDv4)")
    chunk_hash: str = Field(..., description="Hash of the chunk")
    document_id: uuid.UUID = Field(..., description="Unique identifier for the document (UUIDv4)")
    document_hash: str = Field(..., description="Hash of the document")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the chunk")
    related_tags: List[str] = Field(default_factory=list, description="Unique related tags derived from the chunk's tags")
    source: str = Field(..., description="Source of the document")
    document_index: int = Field(..., description="Index of the chunk within the document 0 bound")
    file_type: str = Field(..., description="File type of the document")
    token_count: int = Field(..., description="Token count for the chunk")
    created_at: datetime = Field(..., description="Creation timestamp of the chunk")
    updated_at: datetime = Field(..., description="Last update timestamp of the chunk")
    text: str = Field(..., description="Text content of the chunk")

class VectorSearchResult(BaseSearchResult):
    """Extended search result model with cosine similarity score."""
    cosign_similarity_score: float = Field(..., description="Cosine similarity score for the chunk")

SearchResultType = TypeVar("SearchResultType", bound=BaseSearchResult)
class SearchResponse(GenericModel, Generic[SearchResultType]):
    """Generic response model for search endpoints."""
    metadata: FullMetadata
    data: List[SearchResultType]

class JobBaseData(BaseModel):
    """Job data containing only the job ID."""
    job_id: uuid.UUID = Field(..., description="Job ID (UUIDv4)")

class JobBaseResponse(BaseModel):
    """Base job response containing metadata and only the job ID."""
    metadata: BaseMetadata
    data: JobBaseData

class JobFullData(BaseModel):
    """Job data containing full details.
    
    The following fields are optional:
      - status: The status of the job.
      - document_id: The document ID associated with the job.
      - in_queue_time: Time (in seconds) the job spent in queue.
      - processing_time: Time (in seconds) the job took to process.
      - error_message: Error message if the job failed.
    """
    job_id: uuid.UUID = Field(..., description="Job ID (UUIDv4)")
    created_at: datetime = Field(..., description="Creation timestamp")
    status: Optional[str] = Field(None, description="Status of the job")
    document_id: Optional[uuid.UUID] = Field(None, description="Document ID if available (UUIDv4)")
    in_queue_time: Optional[float] = Field(None, description="Time in queue (seconds)")
    processing_time: Optional[float] = Field(None, description="Processing time (seconds)")
    error_message: Optional[str] = Field(None, description="Error message if job failed")

class JobFullResponse(BaseModel):
    """Full job response containing metadata and complete job details."""
    metadata: BaseMetadata
    data: JobFullData

# ---------------------------
# Helper Function for Response Formatting
# ---------------------------
def format_document_response(results, correlation_id: str = None):
    """Formats search/document responses with full metadata."""
    all_tags = list(chain.from_iterable([result[4] for result in results]))
    related_tags = db.get_related_tags(all_tags)
    metadata = FullMetadata(
        correlation_id=correlation_id,
        token_count=sum(int(result[8]) for result in results),
        chunk_count=len(results),
        related_tags=related_tags if related_tags else []
    )
    data = [
        BaseSearchResult(
            chunk_id=result[0],
            chunk_hash=result[1],
            document_id=result[2],
            document_hash=result[3],
            tags=result[4] if result[4] is not None else [],
            related_tags=db.get_unique_related_tags(result[4]) if result[4] is not None else [],
            source=result[5],
            document_index=int(result[6]),
            file_type=result[7],
            token_count=int(result[8]),
            created_at=result[9],
            updated_at=result[10],
            text=result[11]
        )
        for result in results
    ]
    return {"metadata": metadata, "data": data}

# ---------------------------
# FastAPI Application Initialization
# ---------------------------
app = FastAPI(
    title="Vector Search & Document API",
    description="API for vector search, document processing, chunk management, job status, and health check with unified responses and error formats.",
    version="1.0.0"
)

# ---------------------------
# Middleware
# ---------------------------
@app.middleware("http")
async def add_correlation_id_header(request: Request, call_next):
    """Middleware to add a correlation ID from the header (or generate one) and attach it to request.state."""
    correlation_id = request.headers.get("X-Correlation-Id", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers.append("X-Correlation-Id", correlation_id)
    return response

@app.middleware("http")
async def log_request_data(request: Request, call_next):
    """Middleware to log incoming requests and outgoing responses along with the correlation ID."""
    correlation_id = getattr(request.state, "correlation_id", None)
    logger.info("Request received", extra={"method": request.method, "path": request.url.path, "correlation_id": correlation_id})
    response = await call_next(request)
    logger.info(f"Response sent with status {response.status_code}", extra={"correlation_id": correlation_id})
    return response

# ---------------------------
# Endpoints
# ---------------------------
@app.post(
    "/v1/search",
    response_model=SearchResponse[VectorSearchResult],
    summary="Search the Vector Store by Query",
    description="""
    Generates an embedding from the provided query and searches the vector store.
    Returns document chunks with full metadata including cosine similarity scores.
    """
)
def search_vector_store(request: Request, document_request: DocumentRequest):
    """
    Searches the vector store based on a query string.
    
    Parameters:
      - query: The search query string.
      - max_results: Maximum number of search results to return.
    
    Returns:
      A JSON response containing full metadata and a list of document chunks with cosine similarity scores.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    query = document_request.query.strip()
    max_results = document_request.max_results
    if not query:
        logger.error("Query cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")
    try:
        query_embedding = embedding_function.embed_query(query)
        results = db.similarity_search(query_embedding, max_results)
        if not results:
            logger.error("No results found", extra={"correlation_id": correlation_id, "query": query})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No results found")
        all_tags = list(chain.from_iterable([result[4] for result in results]))
        related_tags = db.get_related_tags(all_tags)
        metadata = FullMetadata(
            correlation_id=correlation_id,
            token_count=sum(int(result[8]) for result in results),
            chunk_count=len(results),
            related_tags=related_tags if related_tags else []
        )
        data = [
            VectorSearchResult(
                chunk_id=result[0],
                chunk_hash=result[1],
                document_id=result[2],
                document_hash=result[3],
                tags=result[4] if result[4] is not None else [],
                related_tags=db.get_unique_related_tags(result[4]) if result[4] is not None else [],
                source=result[5],
                document_index=int(result[6]),
                file_type=result[7],
                token_count=int(result[8]),
                created_at=result[9],
                updated_at=result[10],
                text=result[11],
                cosign_similarity_score=result[12]
            )
            for result in results
        ]
    except Exception as e:
        logger.error("Error searching vector store", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    return {"metadata": metadata, "data": data}

@app.post(
    "/v1/search/tags",
    response_model=SearchResponse[BaseSearchResult],
    summary="Search the Vector Store by Tags using or logic and tsvector",
    description="""
    Searches the vector store using the provided tags.
    Returns document chunks with full metadata (without cosine similarity scores).
    """
)
def search_by_tag(request: Request, document_request: DocumentRequestByTag):
    """
    Searches the vector store based on a list of tags.
    
    Parameters:
      - tags: List of tags to search for.
      - max_results: Maximum number of search results to return.
    
    Returns:
      A JSON response containing full metadata and a list of document chunks (cosine similarity score is null).
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    tags = document_request.tags
    max_results = document_request.max_results
    if not tags:
        logger.error("Tags cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tags cannot be empty")
    tags = [tag.strip() for tag in tags if tag.strip()]
    if not tags:
        logger.error("Tags contain only empty strings after trimming", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tags contain only empty strings after trimming")
    try:
        results = db.search_by_tags(tags, max_results)
        if not results:
            logger.error("Tags not found", extra={"correlation_id": correlation_id, "tags": tags})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tags not found")
        response = format_document_response(results, correlation_id)
    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        logger.error("Error searching vector store", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    return response

@app.post(
    "/v1/document/async",
    response_model=JobBaseResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Document Asynchronously"
)
async def create_document_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    text: str = Form(None),
    max_chunk_size: int = Form(1024)
):
    """
    Creates a document asynchronously.
    
    Accepts either an uploaded file or text (but not both) and enqueues a job for processing.
    Validates the input and returns a job ID in the response.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if file and text:
        logger.error("Cannot provide both file and text", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot provide both file and text")
    if not file and not text:
        logger.error("Must provide file or text", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must provide file or text")
    if max_chunk_size < Config.MIN_CHUNK_SIZE:
        logger.error("Max chunk size too small", extra={"correlation_id": correlation_id, "max_chunk_size": max_chunk_size, "system_min": Config.MIN_CHUNK_SIZE})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"max_chunk_size must be at least {Config.MIN_CHUNK_SIZE}")
    
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
    
    background_tasks.add_task(q.enqueue, "src.worker.process_document", job_id, job_timeout=600)
    logger.info(f"Enqueued job {job_id}", extra={"correlation_id": correlation_id})
    return {
        "metadata": BaseMetadata(correlation_id=correlation_id),
        "data": {"job_id": job_id}
    }

@app.get(
    "/v1/document/hash/{document_hash}", 
    response_model=SearchResponse[BaseSearchResult],
    summary="Get all chunks in a document by document hash"
)
def get_document_by_hash(request: Request, document_hash: str):
    """
    Retrieves all the chunks in a document using the document's hash.
    
    Validates the document hash and returns the chunks.
    """
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

@app.get(
    "/v1/document/{document_id}", 
    response_model=SearchResponse[BaseSearchResult],
    summary="Get all chunks associated with a document by the document ID"
)
def get_document_by_id(request: Request, document_id: str):
    """
    Retrieves all the chunks in a document using the document's id.
    
    Validates the document id and returns the chunks.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not is_valid_uuid(document_id):
        logger.error("Invalid document ID must be UUIDv4", extra={"correlation_id": correlation_id, "document_id": document_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document ID must be UUIDv4")
    try:
        results = db.search_by_document_id(document_id)
        if not results:
            logger.error("Document not found", extra={"correlation_id": correlation_id, "document_id": document_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        response = format_document_response(results, correlation_id)
    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        logger.error("Error fetching document by id", extra={"correlation_id": correlation_id, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    return response

@app.post(
    "/v1/document/preview",
    summary="Preview document by splitting text into chunks"
)
def preview_document(request: Request, document_request: PreviewDocumentRequest):
    """
    Previews a document by splitting the input text into chunks.
    
    Accepts text and optional parameters to compute token counts and tags.
    Returns preview data including chunk metadata.
    """
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
    for i, chunk in enumerate(chunks):
        chunk = add_metadata_to_chunk(chunk, doc_id="preview", document_hash=doc_hash, source_type="preview", index=i, token_count=is_token_count)
        if is_tags:
            chunk.metadata["tags"] = get_chunk_tags(chunk)
    metadata = {
        "correlation_id": correlation_id,
        "chunk_count": len(chunks)
    }
    if is_token_count:
        metadata["token_count"] = sum(chunk.metadata.get("token_count", 0) for chunk in chunks)
    return {
        "metadata": metadata,
        "data": [
            {
                "chunk_hash": chunk.metadata["chunk_hash"],
                "document_hash": chunk.metadata["document_hash"],
                **({"tags": chunk.metadata.get("tags", [])} if is_tags else {}),
                "source": "preview",
                "document_index": chunk.metadata["chunk_index"],
                "file_type": "text",
                **({"token_count": chunk.metadata.get("token_count", 0)} if is_token_count else {}),
                "created_at": chunk.metadata["created_at"],
                "updated_at": chunk.metadata["updated_at"],
                "text": chunk.page_content,
            }
            for chunk in chunks
        ]
    }

@app.get(
    "/v1/chunk/hash/{chunk_hash}",
    response_model=SearchResponse[BaseSearchResult],
    summary="Get Chunk by Hash"
)
def get_chunk_by_hash(request: Request, chunk_hash: str):
    """
    Retrieves a chunk using its hash.
    
    Validates the chunk hash and returns the chunk data in a unified response format.
    """
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

@app.get(
    "/v1/chunk/{chunk_id}",
    response_model=SearchResponse[BaseSearchResult],
    summary="Get Chunk by ID"
)
def get_chunk_by_id(request: Request, chunk_id: str):
    """
    Retrieves a chunk using its ID.
    
    Validates the chunk ID and returns the chunk data in a unified response format.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not is_valid_uuid(chunk_id):
        logger.error("Invalid chunk ID must be UUIDv4", extra={"correlation_id": correlation_id, "chunk_id": chunk_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chunk ID must be UUIDv4")
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

@app.put(
    "/v1/chunk/tags/{chunk_id}",
    response_model=SearchResponse[BaseSearchResult],
    summary="Update Chunk Tags"
)
async def update_document_tags(chunk_id: str, tags_request: UpdateTagsRequest, request: Request):
    """
    Updates the tags for a given chunk.
    
    Validates the new tags and updates the chunk in the database.
    Returns the updated chunk data in a unified response format.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    new_tags = tags_request.tags
    if not new_tags:
        logger.error("Tags cannot be empty", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tags cannot be empty")
    new_tags = [tag.strip() for tag in new_tags if tag.strip()]
    if not new_tags:
        logger.error("Tags contain only empty strings after trimming", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tags contain only empty strings after trimming")
    try:
        is_exists = db.get_chunk_by_id(chunk_id)
        if not is_exists:
            logger.error("Chunk not found", extra={"correlation_id": correlation_id, "chunk_id": chunk_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
        updated_row = db.update_chunk_tags(chunk_id, new_tags)
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

@app.delete(
    "/v1/chunk/{chunk_id}",
    response_model=SearchResponse[BaseSearchResult],
    summary="Soft delete chunk by ID"
)
def delete_chunk_by_id(request: Request, chunk_id: str):
    """
    Soft deletes a chunk using its ID.
    
    Validates the chunk ID, soft deletes the chunk from the database, and returns the deleted chunk data.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not is_valid_uuid(chunk_id):
        logger.error("Invalid chunk ID must be UUIDv4", extra={"correlation_id": correlation_id, "chunk_id": chunk_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chunk ID must be UUIDv4")
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

@app.get(
    "/v1/job/{job_id}",
    response_model=JobFullResponse,
    summary="Get Job Status by ID")
async def get_job_status(request: Request, job_id: str):
    """
    Retrieves the status of a job using its job ID.
    
    Validates the job ID and returns a job response containing status details, queue time, and processing time.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not is_valid_uuid(job_id):
        logger.error("Invalid job ID must be UUIDv4", extra={"correlation_id": correlation_id, "job_id": job_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job ID must be UUIDv4")
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
    if job_status == "failed":
        job_data["error_message"] = error_message
    if job_status == "completed" and updated_at:
        in_queue_time = (start_processing_at - created_at).total_seconds()
        processing_time = (updated_at - start_processing_at).total_seconds()
        job_data["document_id"] = document_id
        job_data["in_queue_time"] = in_queue_time
        job_data["processing_time"] = processing_time
    return {
        "metadata": BaseMetadata(correlation_id=correlation_id),
        "data": job_data
    }

@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Health Check"
)
def health_check(request: Request):
    """
    Performs a health check by pinging the Redis and database services.
    
    Returns a status object indicating whether the system is healthy.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    redis_status = db.ping_redis()
    db_status = db.ping_db()
    
    health_data = HealthData(
        status="ok" if redis_status and db_status else "error",
        redis="ok" if redis_status else "error",
        database="ok" if db_status else "error"
    )
    
    response = HealthResponse(
        metadata=BaseMetadata(correlation_id=correlation_id),  # You can add a correlation_id if needed
        data=health_data
    )
    
    if not (redis_status and db_status):
        logger.error("Health check failed", extra={"redis": redis_status, "database": db_status})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_data.model_dump()
        )
    return response

# ---------------------------
# Custom Exception Handlers
# ---------------------------

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler that returns errors in a unified response format."""
    correlation_id = getattr(request.state, "correlation_id", None)
    error_response = HTTPErrorResponse(
        metadata=BaseMetadata(correlation_id=correlation_id),
        error=HTTPError(message=exc.detail, status_code=exc.status_code)
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(error_response)
    )

@app.exception_handler(RequestValidationError)
async def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom validation error handler that returns errors in a unified response format."""
    correlation_id = getattr(request.state, "correlation_id", None)
    error_response = HTTPErrorResponse(
        metadata=BaseMetadata(correlation_id=correlation_id),
        error=HTTPError(message=str(exc.errors()), status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(error_response)
    )

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler that returns errors in a unified response format."""
    correlation_id = getattr(request.state, "correlation_id", None)
    logger.error("Unhandled exception", extra={"correlation_id": correlation_id, "error": str(exc)})
    error_response = HTTPErrorResponse(
        metadata=BaseMetadata(correlation_id=correlation_id),
        error=HTTPError(message="Internal server error", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(error_response)
    )


# ---------------------------
# Custom OpenAPI Schema
# ---------------------------

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    # Generate the standard schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add custom error model components
    openapi_schema["components"]["schemas"]["BaseMetadata"] = {
        "title": "BaseMetadata",
        "type": "object",
        "properties": {
            "correlation_id": {"title": "Correlation ID", "type": "string"}
        }
    }
    
    openapi_schema["components"]["schemas"]["HTTPError"] = {
        "title": "HTTPError",
        "type": "object",
        "properties": {
            "message": {"title": "Error Message", "type": "string"},
            "status_code": {"title": "Status Code", "type": "integer"}
        }
    }
    
    openapi_schema["components"]["schemas"]["HTTPErrorResponse"] = {
        "title": "HTTPErrorResponse",
        "type": "object",
        "properties": {
            "metadata": {"$ref": "#/components/schemas/BaseMetadata"},
            "error": {"$ref": "#/components/schemas/HTTPError"}
        }
    }

    # Iterate over all endpoints and update the responses:
    for path, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            # Process only HTTP methods
            if method.lower() in ["get", "post", "put", "delete", "patch", "options", "head"]:
                responses = operation.get("responses", {})

                # Remove the default 422 response (validation errors)
                if "422" in responses:
                    del responses["422"]

                # Add a 400 response if not already defined
                if "400" not in responses:
                    responses["400"] = {
                        "description": "Bad Request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HTTPErrorResponse"}
                            }
                        }
                    }
                # Add a 500 response if not already defined
                if "500" not in responses:
                    responses["500"] = {
                        "description": "Internal Server Error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HTTPErrorResponse"}
                            }
                        }
                    }

                operation["responses"] = responses

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)