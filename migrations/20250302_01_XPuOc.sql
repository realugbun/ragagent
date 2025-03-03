-- Enable necessary PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Enum types
CREATE TYPE roles_enum AS ENUM ('user', 'admin');
CREATE TYPE job_status_enum AS ENUM ('created', 'processing', 'completed', 'failed');

-- Tables
CREATE TABLE accounts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE roles (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name roles_enum NOT NULL UNIQUE
);

CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
  role_id UUID REFERENCES roles(id),
  password_hash TEXT NOT NULL,
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  query_limit INT NOT NULL,
  queries_used INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE api_keys (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  key_hash TEXT NOT NULL,
  key_prefix TEXT NOT NULL,
  role_id UUID REFERENCES roles(id),
  expiry_date TIMESTAMP,
  requests_per_minute INT DEFAULT 100,
  requests_per_hour INT DEFAULT 1000,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE collections (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  token_count INT NOT NULL,
  metadata JSONB,
  is_deleted BOOLEAN DEFAULT FALSE,
  deleted_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  is_deleted BOOLEAN DEFAULT FALSE,
  deleted_at TIMESTAMP,
  collection_id UUID REFERENCES collections(id) ON DELETE CASCADE,
  hash CHAR(16) NOT NULL,
  tags text[],
  tags_tvs TSVECTOR,
  token_count INT NOT NULL,
  external_id TEXT,
  source TEXT,
  file_type TEXT,
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_documents_tags_tvs ON documents USING GIN(tags_tvs);

CREATE TABLE chunks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  is_deleted BOOLEAN DEFAULT FALSE,
  deleted_at TIMESTAMP,
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  tags text[],
  tags_tvs TSVECTOR,
  hash CHAR(16) NOT NULL,
  token_count INT NOT NULL,
  document_index INT NOT NULL,
  metadata JSONB,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_chunks_tags_tvs ON chunks USING GIN(tags_tvs);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw(embedding vector_cosine_ops);

CREATE TABLE jobs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  collection_id UUID REFERENCES collections(id) ON DELETE SET NULL,
  document_id UUID,
  status job_status_enum NOT NULL,
  start_processing_at TIMESTAMP,
  text TEXT,
  source TEXT,
  error_message TEXT,
  max_chunk_size INT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE audit (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  path TEXT NOT NULL,
  event_type TEXT NOT NULL CHECK (event_type IN ('login', 'failed_login', 'query', 'role_change')),
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE subscription_types (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE CHECK (name IN ('free', 'pro', 'enterprise')),
  max_tokens INT NOT NULL,
  max_queries INT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE subscriptions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
  subscription_type_id UUID REFERENCES subscription_types(id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK (status IN ('active', 'expired', 'canceled')),
  billing_start TIMESTAMP NOT NULL,
  billing_end TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Lookup Tables
CREATE TABLE accounts_collections (
  account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
  collection_id UUID REFERENCES collections(id) ON DELETE CASCADE,
  PRIMARY KEY (account_id, collection_id)
);

CREATE TABLE users_collections (
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  collection_id UUID REFERENCES collections(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, collection_id)
);

-- Triggers to update tsvector when tags change
CREATE OR REPLACE FUNCTION update_tags_tvs() 
RETURNS TRIGGER AS $$
BEGIN
  NEW.tags_tvs := to_tsvector('english', array_to_string(NEW.tags, ' '));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_documents_tags
BEFORE INSERT OR UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION update_tags_tvs();

CREATE TRIGGER trg_update_chunks_tags
BEFORE INSERT OR UPDATE ON chunks
FOR EACH ROW EXECUTE FUNCTION update_tags_tvs();

-- Function to auto-update updated_at
CREATE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers
DO $$
DECLARE tbl TEXT;
BEGIN
  FOR tbl IN SELECT table_name FROM information_schema.columns WHERE table_schema = 'public' AND column_name = 'updated_at' LOOP
    EXECUTE format('CREATE TRIGGER trg_set_updated_at_%I BEFORE UPDATE ON %I FOR EACH ROW EXECUTE FUNCTION set_updated_at();', tbl, tbl);
  END LOOP;
END $$;

-- keep track of token count in collections when documents are added or removed
CREATE OR REPLACE FUNCTION update_collection_token_count() 
RETURNS TRIGGER AS $$
BEGIN
  -- When a new document is inserted
  IF (TG_OP = 'INSERT') THEN
    UPDATE collections
    SET token_count = (
      SELECT COALESCE(SUM(token_count), 0)
      FROM documents
      WHERE collection_id = NEW.collection_id 
        AND is_deleted = false
    )
    WHERE id = NEW.collection_id;
    RETURN NEW;
  
  ELSIF (TG_OP = 'UPDATE') THEN
    -- If the document changed collections, update both the old and the new one.
    IF NEW.collection_id IS DISTINCT FROM OLD.collection_id THEN
      -- Update old collections token count
      UPDATE collections
      SET token_count = (
        SELECT COALESCE(SUM(token_count), 0)
        FROM documents
        WHERE collection_id = OLD.collection_id 
          AND is_deleted = false
      )
      WHERE id = OLD.collection_id;
      
      -- Update new collection's token count
      UPDATE collections
      SET token_count = (
        SELECT COALESCE(SUM(token_count), 0)
        FROM documents
        WHERE collection_id = NEW.collection_id 
          AND is_deleted = false
      )
      WHERE id = NEW.collection_id;
    
    ELSE
      -- For changes within the same collection (including soft deletion or token_count changes)
      UPDATE collections
      SET token_count = (
        SELECT COALESCE(SUM(token_count), 0)
        FROM documents
        WHERE collection_id = NEW.collection_id 
          AND is_deleted = false
      )
      WHERE id = NEW.collection_id;
    END IF;
    RETURN NEW;
  END IF;
  
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_collection_token_count
AFTER INSERT OR UPDATE ON documents
FOR EACH ROW
EXECUTE FUNCTION update_collection_token_count();
