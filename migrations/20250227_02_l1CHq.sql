-- Create index to optimize search by tags and created_at
-- depends: 20250227_01_R9f7y

CREATE INDEX IF NOT EXISTS cmetadata_created_at_idx ON langchain_pg_embedding 
((cmetadata->>'created_at'));