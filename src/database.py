import re
from psycopg import sql
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
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
                    c.id,
                    c.hash,
                    c.document_id,
                    d.hash AS document_hash,
                    c.tags,
                    d.source,
                    c.document_index,
                    d.file_type,
                    c.token_count,
                    c.created_at,
                    c.updated_at,
                    c.text,
                    1 - (c.embedding <=> %s::vector) AS similarity_score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.is_deleted = FALSE
                ORDER BY similarity_score DESC
                LIMIT %s
            """, (query_embedding, max_results))
            results = cur.fetchall()
    return results

        
def search_by_tags(tags, max_results: int):
    if isinstance(tags, str):
        tags = [tags]
    
    conditions = []
    params = []
    for tag in tags:
        # Use phraseto_tsquery with an explicit cast to text
        conditions.append("c.tags_tvs @@ phraseto_tsquery('english', %s::text)")
        params.append(tag)
    
    where_clause = " OR ".join(conditions)
    
    query = f"""
        SELECT
            c.id,
            c.hash,
            c.document_id,
            d.hash AS document_hash,
            c.tags,
            d.source,
            c.document_index,
            d.file_type,
            c.token_count,
            c.created_at,
            c.updated_at,
            c.text
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.is_deleted = FALSE
          AND ({where_clause})
        ORDER BY c.updated_at DESC
        LIMIT %s
    """
    params.append(max_results)
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def get_unique_related_tags(search_tags: list[str], limit: int = 5, min_frequency: int = 2) -> list[str]:
    """Finds related search tags while filtering out original terms and preventing SQL injection."""

    # Join search terms with ' OR ' to create an OR-based search query
    ts_query = " OR ".join(search_tags)

    sql_query = sql.SQL("""
        WITH query_string AS (
            SELECT format(
                'SELECT tags_tvs FROM chunks WHERE tags_tvs @@ websearch_to_tsquery(''english'', '{}' ) LIMIT 10',
                {}
            ) AS sql_text
        )
        SELECT ts.word, ts.ndoc AS frequency
        FROM ts_stat((SELECT sql_text FROM query_string)) ts
        WHERE ts.word NOT IN (
            SELECT word FROM ts_stat(
                (SELECT format('SELECT to_tsvector(''english'', '{}' )', {}) )
            )
        )
        AND ts.ndoc > %s
        ORDER BY ts.ndoc DESC
        LIMIT %s;
    """).format(sql.Literal(ts_query), sql.Literal(ts_query), sql.Literal(ts_query), sql.Literal(ts_query))


    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_query, (min_frequency, limit))
            related_queries = [row[0] for row in cur.fetchall()]

    return related_queries


def get_related_tags(search_tags: list[str], limit: int = 5, min_frequency: int = 2) -> list[str]:
    """Finds related search tags and returns only what the database provides, safely preventing SQL injection."""

    # Join search terms with ' OR ' to create an OR-based search query
    ts_query = " OR ".join(search_tags) 

    sql_query = sql.SQL("""
        WITH query_string AS (
            SELECT format(
                'SELECT tags_tvs FROM chunks WHERE tags_tvs @@ websearch_to_tsquery(''english'', '{}' ) LIMIT 10',
                {}
            ) AS sql_text
        )
        SELECT ts.word, ts.ndoc AS frequency
        FROM ts_stat((SELECT sql_text FROM query_string)) ts
        WHERE ts.ndoc > %s
        ORDER BY ts.ndoc DESC
        LIMIT %s;
    """).format(sql.Literal(ts_query), sql.Literal(ts_query))

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_query, (min_frequency, limit))  
            related_tags = [row[0] for row in cur.fetchall()]

    return related_tags


#################################################
#                  DOCUMENT                     #
#################################################

def create_document(hash: int, collection_id: str, token_count:int, source: str, file_type: str, external_id: str = None, tags=None):
    if tags is None:
        tags = []

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (hash, collection_id, token_count, external_id, source, file_type, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (hash, collection_id, token_count, external_id, source, file_type, tags))
            document_id = cur.fetchone()[0]
    return document_id

def search_by_document_hash(document_hash: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    c.id,
                    c.hash AS chunk_hash,
                    c.document_id,
                    d.hash AS document_hash,
                    c.tags,
                    d.source,
                    c.document_index,
                    d.file_type,
                    c.token_count,
                    c.created_at,
                    c.updated_at,
                    c.text
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.hash = %s
                AND c.is_deleted = FALSE
                ORDER BY c.document_index ASC
            """, (document_hash,))
            results = cur.fetchall()
    return results

def search_by_document_id(document_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    c.id,
                    c.hash AS chunk_hash,
                    c.document_id,
                    d.hash AS document_hash,
                    c.tags,
                    d.source,
                    c.document_index,
                    d.file_type,
                    c.token_count,
                    c.created_at,
                    c.updated_at,
                    c.text
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.id = %s
                AND c.is_deleted = FALSE
                ORDER BY c.document_index ASC
            """, (document_id,))
            results = cur.fetchall()
    return results

def delete_by_document_id(document_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            # Soft-delete the document
            cur.execute("""
                UPDATE documents
                SET is_deleted = TRUE,
                    deleted_at = NOW()
                WHERE id = %s
            """, (document_id,))
            
            # Soft-delete all associated chunks
            cur.execute("""
                UPDATE chunks
                SET is_deleted = TRUE,
                    deleted_at = NOW()
                WHERE document_id = %s
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
                    c.id,
                    c.hash,
                    c.document_id,
                    d.hash AS document_hash,
                    c.tags,
                    d.source,
                    c.document_index,
                    d.file_type,
                    c.token_count,
                    c.created_at,
                    c.updated_at,
                    c.text
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.hash = %s
                  AND c.is_deleted = FALSE
                ORDER BY c.created_at DESC
            """, (chunk_hash,))
            results = cur.fetchall()
    return results

def get_chunk_by_id(chunk_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    c.id,
                    c.hash AS chunk_hash,
                    c.document_id,
                    d.hash AS document_hash,
                    c.tags,
                    d.source,
                    c.document_index,
                    d.file_type,
                    c.token_count,
                    c.created_at,
                    c.updated_at,
                    c.text
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.id = %s
                  AND c.is_deleted = FALSE
                ORDER BY c.created_at DESC
            """, (chunk_id,))
            results = cur.fetchall()
    return results

def update_chunk_tags(chunk_id: str, new_tags: list[str]):
    # Validate that new_tags is a list of strings.
    if not isinstance(new_tags, list) or not all(isinstance(tag, str) for tag in new_tags):
        raise ValueError("Tags must be a list of strings.")
    
    query = """
    WITH updated AS (
      UPDATE chunks
      SET tags = %s
      WHERE id = %s
      RETURNING *
    )
    SELECT
        updated.id,
        updated.hash AS chunk_hash,
        updated.document_id,
        d.hash AS document_hash,
        updated.tags,
        d.source,
        updated.document_index,
        d.file_type,
        updated.token_count,
        updated.created_at,
        updated.updated_at,
        updated.text
    FROM updated
    JOIN documents d ON updated.document_id = d.id
    """
    
    params = (new_tags, chunk_id)
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            updated_row = cur.fetchone()
    
    return updated_row


def delete_chunk_by_id(chunk_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE chunks
                SET is_deleted = TRUE,
                    deleted_at = NOW()
                WHERE id = %s
            """, (chunk_id,))
            conn.commit()
    return True


def is_chunk_exist_by_id(chunk_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS(
                    SELECT 1
                    FROM chunks
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
                SELECT id
                FROM collections
                WHERE name = %s
            """, (collection_name,))
            result = cur.fetchone()
    return result[0]

#################################################
#                   JOBS                        #
#################################################

def create_job(max_chunk_size: int, collection_id: str, text: str = None, user_id: str = None, source:str = None) -> str:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO jobs (
                        user_id,
                        collection_id,
                        status,
                        text,
                        source,
                        max_chunk_size
                    )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                    user_id,
                    collection_id, 
                    "created", 
                    text,
                    source,
                    max_chunk_size, 
                ))
            job_id = cur.fetchone()[0]
    return job_id

def get_job_status(job_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    status,
                    start_processing_at,
                    document_id,
                    error_message,
                    created_at,
                    updated_at
                FROM jobs
                WHERE id = %s
            """, (job_id,))
            job = cur.fetchone()
    return job