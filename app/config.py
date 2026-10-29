"""
Application settings — every value loaded from environment variables.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Look for .env in both the current dir and the backend dir.
# This makes it work whether you run from project root or from backend/.
_ENV_CANDIDATES = (
    ".env",
    "backend/.env",
    str(Path(__file__).resolve().parent.parent.parent / ".env"),
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_CANDIDATES,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Environment ───────────────────────────────────
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    app_name: str = Field(default="Monitoring API")
    app_version: str = Field(default="1.0.0")

    # ─── Database ──────────────────────────────────────
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="scraping_db")
    db_user: str = Field(default="")
    db_password: str = Field(default="")
    database_url: str = Field(default="")
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_pool_timeout: int = Field(default=30, ge=1, le=300)
    database_pool_recycle: int = Field(default=1800, ge=60)

    @property
    def async_database_url(self) -> str:
        """Use DATABASE_URL if set, otherwise build from parts."""
        if self.database_url:
            return self.database_url
        if not self.db_user:
            raise ValueError(
                "Database credentials not configured. "
                "Set DATABASE_URL or (DB_USER + DB_PASSWORD + DB_HOST + DB_NAME)."
            )
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ─── Redis ─────────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    redis_url: str = Field(default="")
    redis_max_connections: int = Field(default=20, ge=1, le=500)
    redis_socket_timeout: int = Field(default=5, ge=1, le=60)
    redis_connect_timeout: int = Field(default=5, ge=1, le=60)

    @property
    def effective_redis_url(self) -> str:
        """Use REDIS_URL if set, otherwise build from parts."""
        if self.redis_url:
            return self.redis_url
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ─── Monitoring integration ────────────────────────
    monitoring_adapter: str = Field(default="")

    # ─── JWT ───────────────────────────────────────────
    jwt_secret_key: str = Field(default="")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=15, ge=1)
    refresh_token_expire_days: int = Field(default=7, ge=1)
    max_failed_login_attempts: int = Field(default=5, ge=1)
    account_lockout_minutes: int = Field(default=15, ge=1)

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "JWT_SECRET_KEY is missing. "
                "Set it in .env or environment. "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        if len(v) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters. "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        return v

    # ─── CORS ──────────────────────────────────────────
    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000"
    )

    @property
    def allowed_origins_list(self) -> List[str]:
        origins = [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        # Fail-safe: never return empty (would break all CORS)
        return origins or ["http://localhost:5173"]

    # ─── Rate limiting ─────────────────────────────────
    rate_limit_anon_per_minute: int = Field(default=10, ge=1)
    rate_limit_user_per_minute: int = Field(default=100, ge=1)
    rate_limit_paid_per_minute: int = Field(default=1000, ge=1)

    # ─── Security features ─────────────────────────────
    honeypot_enabled: bool = Field(default=True)
    anomaly_detection_enabled: bool = Field(default=True)
    security_headers_enabled: bool = Field(default=True)

    # ─── Logging ───────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="console")

    # ─── Optional external credentials ─────────────────
    skymil_supabase_token: Optional[str] = Field(default=None)
    skymil_supabase_apikey: Optional[str] = Field(default=None)
    skymil_token_expiry: Optional[str] = Field(default=None)

    turnstile_site_key: Optional[str] = Field(default=None)
    turnstile_secret_key: Optional[str] = Field(default=None)

    celery_broker: str = Field(default="")
    celery_backend: str = Field(default="")
        # ─── Public browse limit ────────────────────────────
    # Anonymous users can see this many products per category before
    # being asked to sign in. Set to 0 to require auth for everything.
    public_products_per_category: int = Field(default=200, ge=0, le=10_000)



    # Email
    email_enabled: bool = True
    email_from: str = "noreply@monitoring.local"
    email_from_name: str = "Monitoring System"
    email_reply_to: str | None = None

    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout: int = 10

    email_verify_token_ttl_hours: int = 24
    password_reset_token_ttl_minutes: int = 30

    email_dev_mode: bool = True

    frontend_url: str = "http://localhost:5173"
    api_base_url: str = "http://localhost:8000"


    @property
    def celery_broker_url(self) -> str:
        """Broker defaults to Redis DB 1 (separate from cache DB 0)."""
        return self.celery_broker or self.effective_redis_url.replace("/0", "/1")

    @property
    def celery_result_backend(self) -> str:
        """Results default to Redis DB 2."""
        return self.celery_backend or self.effective_redis_url.replace("/0", "/2")




        # ─── AI / LLM ──────────────────────────────────────
    ai_enabled: bool = Field(default=True)

    llm_provider: str = Field(default="openai")
    llm_fallback_provider: str | None = Field(default="ollama")

    # OpenAI
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_embedding_dims: int = Field(default=1536)

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:8b")
    ollama_embedding_model: str = Field(default="nomic-embed-text")

    # LLM behavior
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=800, ge=1)
    llm_timeout_seconds: int = Field(default=30, ge=1)
    llm_max_retries: int = Field(default=2, ge=0)

    # Cost control
    ai_cache_enabled: bool = Field(default=True)
    ai_cache_ttl_hours: int = Field(default=24, ge=1)
    ai_daily_cost_limit_usd: float = Field(default=1.00, ge=0)
    ai_user_free_credits_per_month: int = Field(default=50, ge=0)

    # Embeddings
    embedding_batch_size: int = Field(default=50, ge=1, le=500)
    embedding_cache_ttl_hours: int = Field(default=168, ge=1)


        # ─── Phase 2 — Embeddings ──────────────────────────
    embedding_provider: str = Field(default="ollama")
    embedding_fallback_enabled: bool = Field(default=False)
    embedding_max_chars: int = Field(default=1000, ge=100, le=8000)
    embedding_similar_cache_ttl_hours: int = Field(default=24, ge=1)
    embedding_max_distance_default: float = Field(default=0.3, ge=0.0, le=2.0)

    # ─── Matching (Phase 3) ─────────────────────────────
    matching_llm_enabled: bool = Field(default=True)
    matching_accept_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    matching_borderline_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    matching_base_cluster_distance: float = Field(default=0.20, ge=0.0, le=2.0)

    destructive_admin_token: str | None = Field(default=None)

    # ─── Recommendations (Phase 4) ─────────────────────
    recs_enabled: bool = Field(default=True)
    recs_llm_enabled: bool = Field(default=False)

    base_group_debug_log: bool = False

    @property
    def ai_available(self) -> bool:
        if not self.ai_enabled:
            return False
        return bool(self.openai_api_key) or self.llm_provider == "ollama"


    @property
    def active_embedding_model(self) -> str:
        """The model name for the currently configured embedding provider."""       
        if self.embedding_provider == "ollama":
            return self.ollama_embedding_model
        return self.openai_embedding_model



@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()