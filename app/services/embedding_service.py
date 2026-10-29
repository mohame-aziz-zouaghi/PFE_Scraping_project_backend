"""
Embedding service — provider-agnostic vector generation.

Also exports the shared text utilities used by the backfill pipeline:
    - build_embedding_text(product)  → canonical text for a product
    - hash_text(text)                → SHA-256 hex digest
    - BackfillStats                  → stats dataclass for backfill runs

Also owns RAG-owned tables:
    - help_embeddings        (per help article, one row)
    - canonical_embeddings   (per canonical product, one row)

Both bulk methods are idempotent upserts. Called from the embedding
queue worker on insert/update hooks, and from startup reconciliation.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import delete, select
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
# Module-level convenience (used by RAG + cache)
# ═══════════════════════════════════════════════════════
async def embed_text(
    text: str,
    *,
    db: Optional[AsyncSession] = None,
    user_id: Optional[UUID] = None,
    endpoint: str = "rag",
) -> Optional[list[float]]:
    """
    Embed a single string. Returns None on any failure.

    Thin wrapper so callers (rag_service, rag_cache, embedding_queue)
    don't need to construct a service, pass a DB session, or handle
    EmbeddingError.
    """
    if not text or not text.strip():
        return None
    try:
        svc = EmbeddingService(db=db, user_id=user_id, endpoint=endpoint)
        return await svc.embed_one(text)
    except EmbeddingError as exc:
        logger.warning("embed_text_failed", error=str(exc)[:200])
        return None
    except Exception as exc:
        logger.warning(
            "embed_text_unhandled", error=str(exc)[:200]
        )
        return None


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
        db: Optional[AsyncSession] = None,
        user_id: Optional[UUID] = None,
        endpoint: str = "embedding",
    ):
        self.db = db
        self.user_id = user_id
        self.endpoint = endpoint

    # ═══════════════════════════════════════════════════════
    # Public API — generic
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
    # Public API — RAG-owned tables (BULK)
    # ═══════════════════════════════════════════════════════
    async def embed_help_articles_bulk(
        self, article_ids: list[UUID]
    ) -> int:
        """
        Bulk-embed N help articles into help_embeddings.

        Handles published + unpublished in one pass:
          • published      → upsert embedding
          • unpublished    → delete existing embedding

        Idempotent. Any existing row for the same article+model
        is replaced in the same transaction.

        Returns the number of embeddings written (unpublished
        articles that had an embedding removed do NOT count).
        """
        if not article_ids or self.db is None:
            return 0

        from app.models.help_article import HelpArticle
        from app.models.help_embedding import HelpEmbedding

        started = time.monotonic()

        # ── 1. Fetch source articles
        rows = list((await self.db.execute(
            select(HelpArticle).where(
                HelpArticle.id.in_(article_ids),
            )
        )).scalars().all())

        if not rows:
            self._log_batch_summary(
                kind="help",
                requested=len(article_ids),
                embedded=0,
                provider="",
                model="",
                elapsed_ms=int(
                    (time.monotonic() - started) * 1000
                ),
            )
            return 0

        # ── 2. Split published / unpublished
        published = [r for r in rows if r.is_published]
        unpublished_ids = [
            r.id for r in rows if not r.is_published
        ]

        # ── 3. Remove embeddings for unpublished articles
        if unpublished_ids:
            await self.db.execute(
                delete(HelpEmbedding).where(
                    HelpEmbedding.article_id.in_(unpublished_ids)
                )
            )

        if not published:
            await self.db.commit()
            self._log_batch_summary(
                kind="help",
                requested=len(article_ids),
                embedded=0,
                provider="",
                model="",
                elapsed_ms=int(
                    (time.monotonic() - started) * 1000
                ),
            )
            return 0

        # ── 4. Build source_text and source_hash
        texts: list[str] = []
        hashes: list[str] = []
        for a in published:
            txt = self._help_source_text(
                title=a.title,
                body=a.body,
                synonyms=getattr(a, "synonyms", None),
            )
            texts.append(txt)
            hashes.append(hash_text(txt))

        # ── 5. Batched embedding call
        try:
            vectors = await self.embed_batch(texts)
        except EmbeddingError as exc:
            await self.db.rollback()
            logger.warning(
                "embed_help_bulk_failed",
                count=len(published),
                error=str(exc)[:200],
            )
            self._log_batch_summary(
                kind="help",
                requested=len(published),
                embedded=0,
                provider=settings.embedding_provider,
                model=settings.active_embedding_model,
                elapsed_ms=int(
                    (time.monotonic() - started) * 1000
                ),
            )
            return 0

        if len(vectors) != len(published):
            logger.warning(
                "embed_help_bulk_partial",
                requested=len(published),
                got=len(vectors),
            )

        ids_to_write = [a.id for a in published[: len(vectors)]]
        hashes_to_write = hashes[: len(vectors)]
        texts_to_write = texts[: len(vectors)]

        # ── 6. Upsert: delete then insert (same model)
        await self.db.execute(
            delete(HelpEmbedding).where(
                HelpEmbedding.article_id.in_(ids_to_write),
                HelpEmbedding.model
                == settings.active_embedding_model,
            )
        )

        self.db.add_all([
            HelpEmbedding(
                article_id=art_id,
                model=settings.active_embedding_model,
                embedding=vec,
                source_text=txt,
                source_hash=src_hash,
            )
            for art_id, vec, txt, src_hash in zip(
                ids_to_write,
                vectors,
                texts_to_write,
                hashes_to_write,
            )
        ])
        await self.db.commit()

        # ── 7. Summary log
        self._log_batch_summary(
            kind="help",
            requested=len(published),
            embedded=len(vectors),
            provider=settings.embedding_provider,
            model=settings.active_embedding_model,
            elapsed_ms=int(
                (time.monotonic() - started) * 1000
            ),
        )
        return len(vectors)

    async def embed_canonical_products_bulk(
    self, canonical_ids: list[UUID]
    ) -> int:
        if not canonical_ids or self.db is None:
            return 0

        from app.models.canonical_embedding import CanonicalEmbedding
        from app.models.canonical_product import CanonicalProduct

        started = time.monotonic()

        rows = list((await self.db.execute(
            select(CanonicalProduct).where(
                CanonicalProduct.id.in_(canonical_ids),
            )
        )).scalars().all())
        if not rows:
            self._log_batch_summary(
                kind="canonical", requested=len(canonical_ids),
                embedded=0, provider="", model="",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return 0

        with_listings = [r for r in rows if (r.listing_count or 0) > 0]
        empty_ids = [r.id for r in rows if (r.listing_count or 0) <= 0]

        if empty_ids:
            await self.db.execute(
                delete(CanonicalEmbedding).where(
                    CanonicalEmbedding.canonical_id.in_(empty_ids)
                )
            )

        if not with_listings:
            await self.db.commit()
            self._log_batch_summary(
                kind="canonical", requested=len(canonical_ids),
                embedded=0, provider="", model="",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return 0

        texts: list[str] = [
            self._canonical_source_text(cp) for cp in with_listings
        ]
        hashes: list[str] = [hash_text(t) for t in texts]

        try:
            vectors = await self.embed_batch(texts)
        except EmbeddingError as exc:
            await self.db.rollback()
            logger.warning(
                "embed_canonical_bulk_failed",
                count=len(with_listings),
                error=str(exc)[:200],
            )
            self._log_batch_summary(
                kind="canonical", requested=len(canonical_ids),
                embedded=0,
                provider=settings.embedding_provider,
                model=settings.active_embedding_model,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return 0

        if len(vectors) != len(with_listings):
            logger.warning(
                "embed_canonical_bulk_partial",
                requested=len(with_listings),
                got=len(vectors),
            )

        # Truncate ALL parallel lists to the same length so the zip
        # can't silently misalign if Ollama returns fewer vectors.
        n = len(vectors)
        cp_to_write = with_listings[:n]
        texts_to_write = texts[:n]
        hashes_to_write = hashes[:n]
        ids_to_write = [cp.id for cp in cp_to_write]

        await self.db.execute(
            delete(CanonicalEmbedding).where(
                CanonicalEmbedding.canonical_id.in_(ids_to_write),
                CanonicalEmbedding.model == settings.active_embedding_model,
            )
        )
        self.db.add_all([
            CanonicalEmbedding(
                canonical_id=cp.id,
                model=settings.active_embedding_model,
                embedding=vec,
                source_text=txt,
                source_hash=h,          # ← the fix
            )
            for cp, vec, txt, h in zip(
                cp_to_write, vectors, texts_to_write, hashes_to_write
            )
        ])
        await self.db.commit()

        self._log_batch_summary(
            kind="canonical",
            requested=len(with_listings),
            embedded=len(vectors),
            provider=settings.embedding_provider,
            model=settings.active_embedding_model,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        return len(vectors)

    # ═══════════════════════════════════════════════════════
    # Public API — RAG-owned tables (SINGLE, thin wrappers)
    # ═══════════════════════════════════════════════════════
    async def embed_help_article(self, article_id: UUID) -> bool:
        """Single-article wrapper. Kept for the hook path."""
        written = await self.embed_help_articles_bulk([article_id])
        return written > 0

    async def embed_canonical_product(
        self, canonical_id: UUID
    ) -> bool:
        """Single-canonical wrapper. Kept for the hook path."""
        written = await self.embed_canonical_products_bulk(
            [canonical_id]
        )
        return written > 0

    # ═══════════════════════════════════════════════════════
    # Text composition for RAG-owned tables
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _help_source_text(
        *, title: str, body: str, synonyms=None,
    ) -> str:
        """
        Build the text embedded for a help article.

        Format:
            {title}
            Synonyms: {syn1} | {syn2} | ...
            {body}

        The synonyms line makes common phrasings ("sign in",
        "log in") match an article titled "How do I create an
        account" at >0.8 cosine similarity. Without it, the same
        question misses the gate at ~0.45.
        """
        parts = [title.strip()]
        if synonyms:
            syn = " | ".join(str(s) for s in synonyms if s)
            if syn:
                parts.append(f"Synonyms: {syn}")
        parts.append(body.strip())
        return "\n".join(parts)

    @staticmethod
    def _canonical_source_text(cp: Any) -> str:
        """
        Canonical chunk text. Mirrors the format the retrieval
        fallback uses, so stored embeddings and search-time
        fallbacks agree.
        """
        name = (getattr(cp, "canonical_name", "") or "").strip()
        site_count = int(getattr(cp, "site_count", 0) or 0)
        cheapest_site = (
            getattr(cp, "cheapest_site", "") or ""
        ).strip()
        cheapest_price = getattr(cp, "cheapest_price", None)

        price_str: Optional[str] = None
        if cheapest_price is not None:
            try:
                price_str = f"{float(cheapest_price):.2f} DT"
            except (TypeError, ValueError):
                price_str = None

        if site_count and cheapest_site and price_str:
            return (
                f"{name} — available at {site_count} sites, "
                f"cheapest at {cheapest_site} for {price_str}"
            )
        if site_count:
            return f"{name} — available at {site_count} sites"
        return name

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
        """
        Batch call to Ollama's /api/embed endpoint.

        Unlike /api/embeddings (single-text legacy), /api/embed
        accepts a list of inputs and returns one vector per
        input, along with `prompt_eval_count` — which we
        forward to CostTracker so llm_usage gets real numbers.
        """
        import httpx

        if not texts:
            return [], {"prompt_tokens": 0, "completion_tokens": 0}

        payload = {
            "model": settings.ollama_embedding_model,
            "input": texts,
        }
        async with httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=settings.llm_timeout_seconds,
        ) as client:
            resp = await client.post("/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()

        embeddings = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings)} vectors for "
                f"{len(texts)} inputs"
            )

        prompt_tokens = int(
            data.get("prompt_eval_count", 0)
            or data.get("prompt_tokens", 0)
        )
        return embeddings, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
        }



        # ═══════════════════════════════════════════════════════
    # Public API — products
    # ═══════════════════════════════════════════════════════
    async def embed_products_bulk(
        self, product_ids: list[UUID]
    ) -> int:
        if not product_ids or self.db is None:
            return 0

        from app.models.product import Product
        from app.repositories.embedding_repo import EmbeddingRepo

        started = time.monotonic()

        rows = list((await self.db.execute(
            select(Product).where(
                Product.id.in_(product_ids),
                Product.is_enabled.is_(True),
            )
        )).scalars().all())

        if not rows:
            self._log_batch_summary(
                kind="product", requested=len(product_ids),
                embedded=0, provider="", model="",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return 0

        texts: list[str] = []
        hashes: list[str] = []
        for p in rows:
            txt = build_embedding_text(p)
            texts.append(txt)
            hashes.append(hash_text(txt))

        try:
            vectors = await self.embed_batch(texts)
        except EmbeddingError as exc:
            await self.db.rollback()
            logger.warning(
                "embed_products_bulk_failed",
                count=len(rows), error=str(exc)[:200],
            )
            self._log_batch_summary(
                kind="product", requested=len(rows), embedded=0,
                provider=settings.embedding_provider,
                model=settings.active_embedding_model,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return 0

        repo = EmbeddingRepo(self.db)
        await repo.upsert_bulk(
            rows=[
                {
                    "product_id": p.id,
                    "embedding": vec,
                    "source_text": txt,
                    "source_hash": h,
                }
                for p, vec, txt, h in zip(
                    rows[: len(vectors)],
                    vectors,
                    texts[: len(vectors)],
                    hashes[: len(vectors)],
                )
            ],
            model=settings.active_embedding_model,
        )
        await self.db.commit()

        self._log_batch_summary(
            kind="product", requested=len(rows),
            embedded=len(vectors),
            provider=settings.embedding_provider,
            model=settings.active_embedding_model,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        return len(vectors)
    
    def _log_batch_summary(
        self,
        *,
        kind: str,
        requested: int,
        embedded: int,
        provider: str,
        model: str,
        elapsed_ms: int,
        ) -> None:
        """
        One line per bulk embed call.

        This replaces the noisy per-request httpx logs with a
        single summary. Called by every embed_*_bulk method.
        """
        failed = max(0, requested - embedded)
        logger.info(
            "embed_batch_summary",
            kind=kind,
            requested=requested,
            embedded=embedded,
            failed=failed,
            provider=provider,
            model=model,
            ms=elapsed_ms,
        )

    # ═══════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _cache_key(text: str) -> str:
        return make_key("embedding", hash_text(text))