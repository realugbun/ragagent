-- 
-- depends: 

CREATE TABLE IF NOT EXISTS jobs (
    job_id          UUID        PRIMARY KEY,
    tenant_id       TEXT,
    document_id     UUID,
    file_path       TEXT,
    max_chunk_size  INT         DEFAULT 1024,
    status          TEXT        DEFAULT 'pending',
    collection_id   UUID,
    text            TEXT,
    created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);
CREATE INDEX IF NOT EXISTS jobs_tenant_id_idx ON jobs (status);