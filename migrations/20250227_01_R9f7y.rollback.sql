-- 
-- depends: 20250226_01_Ibvx7

DROP INDEX IF EXISTS embedding_hnsw_idx;

ALTER TABLE langchain_pg_embedding
ALTER COLUMN embedding TYPE VECTOR;