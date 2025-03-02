import json
from psycopg_pool import ConnectionPool
import uuid
from dotenv import load_dotenv
from langchain_postgres import PGVector
from redis import Redis

from src.config import Config

load_dotenv()

conninfo = f"dbname={Config.DB_NAME} user={Config.DB_USER} password={Config.DB_PASSWORD} host={Config.DB_HOST} port={Config.DB_PORT}"

pool = ConnectionPool(
    conninfo=conninfo,
    min_size=1,
    max_size=10,
    )

#################################################
#                  REDIS                        #
#################################################

def get_redis():
    return Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        password=Config.REDIS_PASSWORD,
        db=Config.REDIS_DB,
    )

def ping_redis() -> bool:
    try:
        redis_client: Redis = get_redis()
        return redis_client.ping()
    except Exception:
        return False
    
#################################################
#                  POSTGRES                     #
#################################################

def get_pgvector_db(embedding_function):
    return PGVector(
    connection=Config.DB_CONNECTION_STRING,
    embeddings=embedding_function,
    collection_name="docs",
    create_extension=False,
    )

def get_db():
    return pool.connection()

def ping_db() -> bool:
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result == (1,)
    except Exception:
        return False
    
#################################################
#                   SEARCH                      #
#################################################

def similarity_search(query_embedding, max_results):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    cmetadata->>'chunk_hash' as chunk_hash,
                    cmetadata->>'document_id' as document_id,
                    cmetadata->'document_hash' as document_hash,
                    cmetadata->'tags' as tags,
                    cmetadata->>'source' as source,
                    cmetadata->>'chunk_index' as chunk_index,
                    cmetadata->>'start_index' as start_index,
                    cmetadata->>'source_type' as source_type,
                    cmetadata->>'token_count' as token_count,
                    cmetadata->>'created_at' as created_at,
                    cmetadata->>'updated_at' as updated_at,
                    document,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM langchain_pg_embedding
                ORDER BY similarity_score DESC
                LIMIT %s
            """, (query_embedding, max_results))
            results = cur.fetchall()
    return results

def search_by_tags(tags, max_results: int):
    if isinstance(tags, str):
        tags = [tags]
    
    tag_conditions = ["tag.value ILIKE %s" for _ in tags]
    query = """
        SELECT
            id,
            cmetadata->>'chunk_hash' AS chunk_hash,
            cmetadata->>'document_id' AS document_id,
            cmetadata->>'document_hash' AS document_hash,
            cmetadata->>'tags' AS tags,
            cmetadata->>'source' AS source,
            cmetadata->>'chunk_index' AS chunk_index,
            cmetadata->>'start_index' AS start_index,
            cmetadata->>'source_type' AS source_type,
            cmetadata->>'token_count' AS token_count,
            cmetadata->>'created_at' AS created_at,
            cmetadata->>'updated_at' AS updated_at,
            document
        FROM langchain_pg_embedding,
             LATERAL jsonb_array_elements_text(cmetadata->'tags') AS tag(value)
        WHERE {}
        ORDER BY cmetadata->>'created_at' DESC
        LIMIT %s
    """.format(" OR ".join(tag_conditions))
    
    params = [f"%{tag}%" for tag in tags] + [max_results]
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

#################################################
#                  DOCUMENT                     #
#################################################

def search_by_document_hash(document_hash):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    cmetadata->>'chunk_hash' as chunk_hash,
                    cmetadata->>'document_id' as document_id,
                    cmetadata->'document_hash' as document_hash,
                    cmetadata->'tags' as tags,
                    cmetadata->>'source' as source,
                    cmetadata->>'chunk_index' as chunk_index,
                    cmetadata->>'start_index' as start_index,
                    cmetadata->>'source_type' as source_type,
                    cmetadata->>'token_count' as token_count,
                    cmetadata->>'created_at' as created_at,
                    cmetadata->>'updated_at' as updated_at,
                    document
                FROM langchain_pg_embedding
                WHERE cmetadata->>'document_hash' = %s
                ORDER BY cmetadata->>'created_at' DESC
            """, (document_hash,))
            results = cur.fetchall()
    return results

def search_by_document_id(document_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    cmetadata->>'chunk_hash' as chunk_hash,
                    cmetadata->>'document_id' as document_id,
                    cmetadata->'document_hash' as document_hash,
                    cmetadata->'tags' as tags,
                    cmetadata->>'source' as source,
                    cmetadata->>'chunk_index' as chunk_index,
                    cmetadata->>'start_index' as start_index,
                    cmetadata->>'source_type' as source_type,
                    cmetadata->>'token_count' as token_count,
                    cmetadata->>'created_at' as created_at,
                    cmetadata->>'updated_at' as updated_at,
                    document
                FROM langchain_pg_embedding
                WHERE cmetadata->>'document_id' = %s
                ORDER BY cmetadata->>'created_at' DESC
            """, (document_id,))
            results = cur.fetchall()
    return results

def delete_by_document_id(document_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM langchain_pg_embedding
                WHERE cmetadata->>'document_id' = %s
            """, (document_id,))
            conn.commit()
    return True

#################################################
#                  CHUNK                        #
#################################################

def get_chunk_by_hash(chunk_hash: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    cmetadata->>'chunk_hash' as chunk_hash,
                    cmetadata->>'document_id' as document_id,
                    cmetadata->'document_hash' as document_hash,
                    cmetadata->'tags' as tags,
                    cmetadata->>'source' as source,
                    cmetadata->>'chunk_index' as chunk_index,
                    cmetadata->>'start_index' as start_index,
                    cmetadata->>'source_type' as source_type,
                    cmetadata->>'token_count' as token_count,
                    cmetadata->>'created_at' as created_at,
                    cmetadata->>'updated_at' as updated_at,
                    document
                FROM langchain_pg_embedding
                WHERE cmetadata->>'chunk_hash' = %s
                ORDER BY cmetadata->>'created_at' DESC
            """, (chunk_hash,))
            results = cur.fetchall()
    return results

def get_chunk_by_id(chunk_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    cmetadata->>'chunk_hash' as chunk_hash,
                    cmetadata->>'document_id' as document_id,
                    cmetadata->'document_hash' as document_hash,
                    cmetadata->'tags' as tags,
                    cmetadata->>'source' as source,
                    cmetadata->>'chunk_index' as chunk_index,
                    cmetadata->>'start_index' as start_index,
                    cmetadata->>'source_type' as source_type,
                    cmetadata->>'token_count' as token_count,
                    cmetadata->>'created_at' as created_at,
                    cmetadata->>'updated_at' as updated_at,
                    document
                FROM langchain_pg_embedding
                WHERE id = %s
                ORDER BY cmetadata->>'created_at' DESC
            """, (chunk_id,))
            results = cur.fetchall()
    return results

def delete_chunk_by_id(chunk_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM langchain_pg_embedding
                WHERE id = %s
            """, (chunk_id,))
            conn.commit()
    return True


def update_chunk_tags(chunk_id: str, new_tags: list[str]):
    # Ensure the tags are a valid list of strings
    if not isinstance(new_tags, list) or not all(isinstance(tag, str) for tag in new_tags):
        raise ValueError("Tags must be a list of strings.")
    
    # Convert the new tags list to a JSONB array format for the database
    tags_jsonb = json.dumps(new_tags)
    
    query = """
    UPDATE langchain_pg_embedding
    SET cmetadata = jsonb_set(
            jsonb_set(cmetadata, '{tags}', %s::jsonb),
            '{updated_at}', to_jsonb(CURRENT_TIMESTAMP), true
        )
    WHERE id = %s
    RETURNING
        id,
        cmetadata->>'chunk_hash' as chunk_hash,
        cmetadata->>'document_id' as document_id,
        cmetadata->'document_hash' as document_hash,
        cmetadata->'tags' as tags,
        cmetadata->>'source' as source,
        cmetadata->>'chunk_index' as chunk_index,
        cmetadata->>'start_index' as start_index,
        cmetadata->>'source_type' as source_type,
        cmetadata->>'token_count' as token_count,
        cmetadata->>'created_at' as created_at,
        cmetadata->>'updated_at' as updated_at,
        document
    """
    
    params = [tags_jsonb, chunk_id]
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            updated_row = cur.fetchone()
    
    return updated_row

def is_chunk_exist_by_id(chunk_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS(
                    SELECT 1
                    FROM langchain_pg_embedding
                    WHERE id = %s
                )
            """, (chunk_id,))
            result = cur.fetchone()
    return result[0]

#################################################
#                  COLLECTION                   #
#################################################

def get_collection_id(collection_name: str) -> str:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT uuid
                FROM langchain_pg_collection
                WHERE name = %s
            """, (collection_name,))
            result = cur.fetchone()
    return result[0]

#################################################
#                   JOBS                        #
#################################################

def create_job(max_chunk_size: int, collection_id: str, text: str = None, tenant_id: str = "default", file_path:str = None) -> str:
    job_id = str(uuid.uuid4())
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO jobs (
                        job_id,
                        tenant_id,
                        max_chunk_size,
                        status,
                        collection_id,
                        text,
                        file_path
                    )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING job_id
            """, (
                    job_id, 
                    tenant_id,
                    max_chunk_size, 
                    "created", 
                    collection_id, 
                    text,
                    file_path
                ))
    return job_id