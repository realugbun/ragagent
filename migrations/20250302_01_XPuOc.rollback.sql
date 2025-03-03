-- Drop triggers
DROP TRIGGER IF EXISTS trg_update_documents_tags ON documents;
DROP TRIGGER IF EXISTS trg_update_chunks_tags ON chunks;
DROP FUNCTION IF EXISTS update_tags_tvs;

-- Drop updated_at auto-update triggers
DO $$
DECLARE tbl TEXT;
BEGIN
  FOR tbl IN SELECT table_name FROM information_schema.columns WHERE table_schema = 'public' AND column_name = 'updated_at' LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS trg_set_updated_at_%I ON %I;', tbl, tbl);
  END LOOP;
END $$;
DROP FUNCTION IF EXISTS set_updated_at;

-- Drop indexes
DROP INDEX IF EXISTS idx_documents_tags_tvs;
DROP INDEX IF EXISTS idx_chunks_tags_tvs;
DROP INDEX IF EXISTS idx_chunks_embedding;

-- Drop lookup tables
DROP TABLE IF EXISTS accounts_collections;
DROP TABLE IF EXISTS users_collections;

-- Drop main tables
DROP TABLE IF EXISTS audit;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS documents;
DROP TABLE IF EXISTS collections;
DROP TABLE IF EXISTS api_keys;
DROP TABLE IF EXISTS roles;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS subscription_types;

-- Drop enum types
DROP TYPE IF EXISTS roles_enum;
DROP TYPE IF EXISTS job_status_enum;

-- Drop extensions
DROP EXTENSION IF NOT EXISTS "vector";
DROP EXTENSION IF NOT EXISTS "pgcrypto";
DROP EXTENSION IF NOT EXISTS "uuid-ossp";
