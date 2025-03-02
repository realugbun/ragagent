-- 
-- depends: 20250226_01_Ibvx7

ALTER TABLE langchain_pg_embedding
ALTER COLUMN embedding TYPE VECTOR(1536);

CREATE INDEX IF NOT EXISTS embedding_hnsw_idx ON langchain_pg_embedding 
USING hnsw (embedding vector_cosine_ops);