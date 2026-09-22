-- ═══════════════════════════════════════════════════════
-- Migration 005 — LLM usage / cost tracking
-- Safe to run on existing DB (uses IF NOT EXISTS).
-- ═══════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS llm_usage (
    id                 BIGSERIAL PRIMARY KEY,

    user_id            UUID REFERENCES users(id) ON DELETE SET NULL,

    endpoint           VARCHAR(100) NOT NULL,
    provider           VARCHAR(50)  NOT NULL,
    model              VARCHAR(100) NOT NULL,

    prompt_tokens      INTEGER      NOT NULL DEFAULT 0,
    completion_tokens  INTEGER      NOT NULL DEFAULT 0,
    total_tokens       INTEGER      NOT NULL DEFAULT 0,

    cost_cents         NUMERIC(10,4) NOT NULL DEFAULT 0,
    latency_ms         INTEGER      NOT NULL DEFAULT 0,

    cache_hit          BOOLEAN      NOT NULL DEFAULT FALSE,
    success            BOOLEAN      NOT NULL DEFAULT TRUE,
    error_message      VARCHAR(500),

    prompt_version     VARCHAR(20),
    request_id         VARCHAR(64),

    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_llm_usage_user_created
    ON llm_usage (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_llm_usage_endpoint_created
    ON llm_usage (endpoint, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_llm_usage_created_at
    ON llm_usage (created_at DESC);
