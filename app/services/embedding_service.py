"""
Embedding service — provider-agnostic vector generation.

Also exports the shared text utilities used by the backfill pipeline:
    - build_embedding_text(product)  → canonical text for a product
    - hash_text(text)                → SHA-256 hex digest
    - BackfillStats                  → stats dataclass for backfill runs

No imports from embedding_backfill_service — this is the low-level module.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.cache import Cache, make_key
from app.core.cost_tracker import CostTracker
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when every embedding provider fails."""


# ═══════════════════════════════════════════════════════
# Text composition (shared by service + backfill)
# ═══════════════════════════════════════════════════════
def build_embedding_text(product: Any) -> str:
    """
    Canonical text used to embed a product.

    Order: brand | name | category | short_description.
    Whitespace collapsed, capped at settings.embedding_max_chars.
    """
    parts: list[str] = []

    brand = getattr(product, "brand", None)
    if brand:
        parts.append(str(brand).strip())

    name = getattr(product, "name", None)
    if name:
        parts.append(str(name).strip())

    category = getattr(product, "category", None)
    if category:
        parts.append(str(category).strip())

    short = getattr(product, "short_description", None)
    if short:
        try:
            import json as _json
            short = _json.dumps(short, ensure_ascii=False)
        except Exception:
            short = str(short)
        parts.append(str(short).strip())

    text = " | ".join(p for p in parts if p)
    text = " ".join(text.split())
    max_chars = getattr(settings, "embedding_max_chars", 1000)
    return text[:max_chars]


def hash_text(text: str) -> str:
    """SHA-256 hex digest of the text (64 chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════
# Stats dataclass (returned by backfill service)
# ═══════════════════════════════════════════════════════
@dataclass
class BackfillStats:
    total_candidates: int = 0
    processed: int = 0
    embedded: int = 0
    skipped: int = 0
    failed: int = 0

    def as_dict(self) -> dict:
        return {
            "total_candidates": self.total_candidates,
            "processed": self.processed,
            "embedded": self.embedded,
            "skipped": self.skipped,
            "failed": self.failed,
        }


# ═══════════════════════════════════════════════════════
# EmbeddingService
# ═══════════════════════════════════════════════════════
class EmbeddingService:
    def __init__(
        self,
        *,
        db: Optional[AsyncSession] = None,
        user_id: Optional[UUID] = None,
        endpoint: str = "embedding",
    ):
        self.db = db
        self.user_id = user_id
        self.endpoint = endpoint

    # ═══════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════
    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Preserves order. Uses cache per item."""
        if not texts:
            return []

        vectors: list[Optional[list[float]]] = [None] * len(texts)
        misses: list[tuple[int, str]] = []

        # ── 1. Cache lookup
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = await Cache.get(key)
            if cached is not None:
                vectors[i] = cached
            else:
                misses.append((i, text))

        if not misses:
            logger.debug("embedding_all_cached", count=len(texts))
            return [v for v in vectors if v is not None]

        # ── 2. Batch call for misses
        batch_size = settings.embedding_batch_size
        for start in range(0, len(misses), batch_size):
            chunk = misses[start : start + batch_size]
            chunk_texts = [t for _, t in chunk]

            vecs, provider, model, usage = await self._embed_provider(
                chunk_texts
            )

            # ── 3. Store in cache
            for (i, text), vec in zip(chunk, vecs):
                vectors[i] = vec
                await Cache.set(
                    self._cache_key(text),
                    vec,
                    settings.embedding_cache_ttl_hours * 3600,
                )

            # ── 4. Record cost
            if self.db:
                await CostTracker.record(
                    self.db,
                    endpoint=self.endpoint,
                    provider=provider,
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    latency_ms=0,
                    user_id=self.user_id,
                )

        return [v for v in vectors if v is not None]

    # ═══════════════════════════════════════════════════════
    # Provider calls
    # ═══════════════════════════════════════════════════════
    async def _embed_provider(
    self, texts: list[str]
) -> tuple[list[list[float]], str, str, dict]:
        """
        Dispatch to the configured embedding provider.

        - EMBEDDING_PROVIDER=openai → OpenAI only (fallback if enabled)
        - EMBEDDING_PROVIDER=ollama → Ollama only
        """
        provider = settings.embedding_provider.lower()

        if provider == "openai":
            if not settings.openai_api_key:
                raise EmbeddingError(
                    "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is empty"
                )
            try:
                started = time.monotonic()
                vecs, usage = await self._embed_openai(texts)
                logger.debug(
                    "embedding_openai_ok",
                    count=len(texts),
                    ms=int((time.monotonic() - started) * 1000),
                )
                return vecs, "openai", settings.openai_embedding_model, usage
            except Exception as e:
                if not settings.embedding_fallback_enabled:
                    raise EmbeddingError(f"OpenAI embedding failed: {e}")
                logger.warning("embedding_openai_failed", error=str(e))

        # Ollama path (default when EMBEDDING_PROVIDER=ollama)
        try:
            started = time.monotonic()
            vecs, usage = await self._embed_ollama(texts)
            logger.debug(
                "embedding_ollama_ok",
                count=len(texts),
                ms=int((time.monotonic() - started) * 1000),
            )
            return vecs, "ollama", settings.ollama_embedding_model, usage
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding failed: {e}")

    async def _embed_openai(
        self, texts: list[str]
    ) -> tuple[list[list[float]], dict]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )
        resp = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )
        ordered = sorted(resp.data, key=lambda d: d.index)
        vecs = [d.embedding for d in ordered]
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": 0,
        }
        return vecs, usage

    async def _embed_ollama(
        self, texts: list[str]
    ) -> tuple[list[list[float]], dict]:
        import ollama

        client = ollama.AsyncClient(host=settings.ollama_base_url)
        vecs: list[list[float]] = []
        prompt_tokens = 0
        for t in texts:
            resp = await client.embeddings(
                model=settings.ollama_embedding_model, prompt=t
            )
            vecs.append(resp["embedding"])
            prompt_tokens += resp.get("prompt_eval_count", 0)
        return vecs, {"prompt_tokens": prompt_tokens, "completion_tokens": 0}

    # ═══════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _cache_key(text: str) -> str:
        return make_key("embedding", hash_text(text))