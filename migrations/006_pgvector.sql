-- ═══════════════════════════════════════════════════════
-- Migration 006 — pgvector + product_embeddings
-- Safe to run on existing DB (idempotent).
-- ═══════════════════════════════════════════════════════

-- ─── 1. Enable the pgvector extension ───────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── 2. product_embeddings table ────────────────────────
CREATE TABLE IF NOT EXISTS product_embeddings (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id   UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    embedding    vector(1536) NOT NULL,
    model        VARCHAR(100) NOT NULL,
    source_text  TEXT NOT NULL,
    source_hash  VARCHAR(64) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (product_id, model)
);

-- ─── 3. HNSW index for fast cosine similarity ───────────
-- Created separately because IF NOT EXISTS on index with HNSW
-- can fail on older pgvector versions. Guard with a DO block.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'ix_product_embeddings_hnsw'
    ) THEN
        CREATE INDEX ix_product_embeddings_hnsw
            ON product_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
    END IF;
END$$;

-- ─── 4. Lookup index for product_id ─────────────────────
CREATE INDEX IF NOT EXISTS ix_product_embeddings_product
    ON product_embeddings (product_id);

-- ─── 5. products: embedding tracking columns ────────────
ALTER TABLE products
    ADD COLUMN IF NOT EXISTS embedding_status VARCHAR(20) DEFAULT 'pending';

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS last_embedded_at TIMESTAMPTZ;

-- Backfill existing rows so the status is meaningful
UPDATE products
    SET embedding_status = 'pending'
    WHERE embedding_status IS NULL;

-- Partial index — only enabled products with pending/ready status
CREATE INDEX IF NOT EXISTS ix_products_embedding_status
    ON products (embedding_status, is_enabled)
    WHERE is_enabled = TRUE;

-- ─── 6. Sanity check ────────────────────────────────────
-- Should return one row: 'vector'
SELECT extname FROM pg_extension WHERE extname = 'vector';
