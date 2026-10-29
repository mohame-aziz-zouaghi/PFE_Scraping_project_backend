# app/services/rag_retrieval.py
"""
RAG retrieval — three pgvector queries + direct help lookup.

Two public entry points:

    retrieve(...)          — the standard three-index retrieval,
                             used for product / price / availability
                             questions.

    find_help_article(...) — high-confidence lookup in the
                             help_articles corpus only. Used to
                             answer FAQ-class questions directly
                             from the stored article body, without
                             the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.core.rag_config import (
    RAG,
    RAG_SELLABLE_AVAILABILITY,
)

logger = get_logger(__name__)


ChunkKind = Literal["product", "canonical", "help"]


@dataclass
class RetrievedChunk:
    chunk_id: str
    kind: ChunkKind
    text: str
    cosine: float
    rerank_score: float = 0.0
    category: Optional[str] = None
    product_id: Optional[UUID] = None
    canonical_id: Optional[UUID] = None
    article_id: Optional[UUID] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    top_score: float
    had_any_hits: bool
    category_hint: Optional[str] = None


@dataclass
class HelpMatch:
    """A single direct help-article match."""
    article_id: UUID
    slug: str
    title: str
    body: str
    category: str
    locale: str
    cosine: float


class RetrievalService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model = settings.active_embedding_model

    # ═══════════════════════════════════════════════════
    # Public — direct help lookup (used before the LLM path)
    # ═══════════════════════════════════════════════════
    async def find_help_article(
        self,
        *,
        query_embedding: list[float],
        locale: str = "en",
        min_similarity: float = 0.72,
    ) -> Optional[HelpMatch]:
        """
        Look up a single help_article whose embedding is a
        very strong match for the query. Returns the article's
        body so the caller can answer WITHOUT the LLM.

        `min_similarity` is intentionally high (0.72) — this
        path is meant to fire only when we're confident the
        user is asking one of the curated FAQ questions. A
        weaker match falls through to the standard pipeline.

        Returns None if no match exceeds the threshold.
        """
        sql = text(
            """
            SELECT
                ha.id,
                ha.slug,
                ha.title,
                ha.body,
                ha.category,
                ha.locale,
                1 - (he.embedding <=> CAST(:vec AS vector)) AS sim
            FROM help_embeddings he
            JOIN help_articles ha ON ha.id = he.article_id
            WHERE he.model = :model
              AND ha.is_published = TRUE
              AND ha.locale = :locale
            ORDER BY he.embedding <=> CAST(:vec AS vector)
            LIMIT 1
            """
        )
        try:
            row = (await self.db.execute(sql, {
                "vec": str(query_embedding),
                "model": self.model,
                "locale": locale,
            })).first()
        except Exception as exc:
            logger.warning(
                "rag_find_help_failed", error=str(exc)[:200]
            )
            return None

        if row is None or row.sim is None:
            return None
        if float(row.sim) < min_similarity:
            return None

        return HelpMatch(
            article_id=row.id,
            slug=row.slug,
            title=row.title,
            body=row.body,
            category=row.category,
            locale=row.locale,
            cosine=float(row.sim),
        )

    # ═══════════════════════════════════════════════════
    # Public — standard retrieval (products + canonical + help)
    # ═══════════════════════════════════════════════════
    async def retrieve(
        self,
        *,
        query_embedding: list[float],
        locale: str = "en",
        product_top_k: int = RAG.TOP_K_PRODUCT_CHUNKS,
        canonical_top_k: int = RAG.TOP_K_CANONICAL_CHUNKS,
        help_top_k: int = RAG.TOP_K_HELP_CHUNKS,
    ) -> RetrievalResult:
        """
        Run the three indices SEQUENTIALLY.

        Why sequential: AsyncSession does not support concurrent
        operations on the same session — running three queries
        in parallel with asyncio.create_task produced
        "concurrent operations are not permitted" errors and
        silently dropped the canonical and help chunks.

        With HNSW indexing each query completes in 5-20ms, so
        three sequential queries = 15-60ms total. That's fast
        enough; parallelism here was a false optimization.
        """
        if not query_embedding:
            return RetrievalResult(
                chunks=[], top_score=0.0, had_any_hits=False
            )

        product_chunks = await self._retrieve_products(
            query_embedding=query_embedding, limit=product_top_k
        )
        canonical_chunks = await self._retrieve_canonicals(
            query_embedding=query_embedding, limit=canonical_top_k
        )
        help_chunks = await self._retrieve_help(
            query_embedding=query_embedding,
            limit=help_top_k,
            locale=locale,
        )

        all_chunks = product_chunks + canonical_chunks + help_chunks
        if not all_chunks:
            return RetrievalResult(
                chunks=[], top_score=0.0, had_any_hits=False
            )

        reranked = self._rerank(all_chunks)
        deduped = self._dedupe(reranked)
        selected = deduped[: RAG.MAX_CONTEXT_CHUNKS]
        category_hint = self._pick_category_hint(selected)
        top_score = max(c.cosine for c in all_chunks)

        return RetrievalResult(
            chunks=selected,
            top_score=top_score,
            had_any_hits=True,
            category_hint=category_hint,
        )

    # ═══════════════════════════════════════════════════
    # Index 1: products
    # ═══════════════════════════════════════════════════
    async def _retrieve_products(
        self,
        *,
        query_embedding: list[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        sql = text(
            """
            SELECT
                p.id                AS product_id,
                p.name              AS name,
                p.brand             AS brand,
                p.category          AS category,
                p.site              AS site,
                p.price             AS price,
                p.discount_price    AS discount_price,
                p.availability      AS availability,
                p.main_image        AS main_image,
                p.url               AS url,
                p.last_updated_at   AS last_updated_at,
                pe.source_text      AS source_text,
                1 - (pe.embedding <=> CAST(:vec AS vector)) AS sim
            FROM product_embeddings pe
            JOIN products p ON p.id = pe.product_id
            WHERE pe.model = :model
              AND p.is_enabled = TRUE
              AND p.availability = ANY(:sellable)
            ORDER BY pe.embedding <=> CAST(:vec AS vector)
            LIMIT :lim
            """
        )
        try:
            rows = (await self.db.execute(sql, {
                "vec": str(query_embedding),
                "model": self.model,
                "sellable": list(RAG_SELLABLE_AVAILABILITY),
                "lim": limit,
            })).all()
        except Exception as exc:
            logger.warning(
                "rag_retrieve_products_failed", error=str(exc)[:200]
            )
            return []

        out: list[RetrievedChunk] = []
        for r in rows:
            if r.sim is None:
                continue
            text_body = _truncate(
                (r.source_text or _product_text(
                    r.name, r.brand, r.category, r.price,
                )),
                RAG.MAX_CHUNK_CHARS,
            )
            out.append(RetrievedChunk(
                chunk_id=f"product:{r.product_id}",
                kind="product",
                text=text_body,
                cosine=float(r.sim),
                category=r.category,
                product_id=r.product_id,
                meta={
                    "name": r.name,
                    "brand": r.brand,
                    "site": r.site,
                    "price": _to_float(r.price),
                    "discount_price": _to_float(r.discount_price),
                    "availability": r.availability,
                    "main_image": r.main_image,
                    "url": r.url,
                    "last_updated_at": r.last_updated_at,
                },
            ))
        return out

    # ═══════════════════════════════════════════════════
    # Index 2: canonical products
    # ═══════════════════════════════════════════════════
    async def _retrieve_canonicals(
        self,
        *,
        query_embedding: list[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        sql = text(
            """
            SELECT
                cp.id                   AS canonical_id,
                cp.canonical_name       AS canonical_name,
                cp.canonical_brand      AS canonical_brand,
                cp.canonical_category   AS canonical_category,
                cp.listing_count        AS listing_count,
                cp.site_count           AS site_count,
                cp.cheapest_price       AS cheapest_price,
                cp.cheapest_site        AS cheapest_site,
                ce.source_text          AS source_text,
                1 - (ce.embedding <=> CAST(:vec AS vector)) AS sim
            FROM canonical_embeddings ce
            JOIN canonical_products cp ON cp.id = ce.canonical_id
            WHERE ce.model = :model
              AND cp.listing_count > 0
              AND EXISTS (
                  SELECT 1 FROM products p
                  WHERE p.canonical_id = cp.id
                    AND p.is_enabled = TRUE
                    AND p.availability = ANY(:sellable)
              )
            ORDER BY ce.embedding <=> CAST(:vec AS vector)
            LIMIT :lim
            """
        )
        try:
            rows = (await self.db.execute(sql, {
                "vec": str(query_embedding),
                "model": self.model,
                "lim": limit,
                "sellable": list(RAG_SELLABLE_AVAILABILITY),
            })).all()
        except Exception as exc:
            logger.warning(
                "rag_retrieve_canonicals_failed", error=str(exc)[:200]
            )
            return []

        out: list[RetrievedChunk] = []
        for r in rows:
            if r.sim is None:
                continue
            text_body = _truncate(
                r.source_text or _canonical_text(
                    r.canonical_name, r.site_count,
                    r.cheapest_site, r.cheapest_price,
                ),
                RAG.MAX_CHUNK_CHARS,
            )
            out.append(RetrievedChunk(
                chunk_id=f"canonical:{r.canonical_id}",
                kind="canonical",
                text=text_body,
                cosine=float(r.sim),
                category=r.canonical_category,
                canonical_id=r.canonical_id,
                meta={
                    "canonical_name": r.canonical_name,
                    "brand": r.canonical_brand,
                    "listing_count": r.listing_count,
                    "site_count": r.site_count,
                    "cheapest_price": _to_float(r.cheapest_price),
                    "cheapest_site": r.cheapest_site,
                },
            ))
        return out

    # ═══════════════════════════════════════════════════
    # Index 3: help articles
    # ═══════════════════════════════════════════════════
    async def _retrieve_help(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        locale: str = "en",
    ) -> list[RetrievedChunk]:
        sql = text(
            """
            SELECT
                ha.id           AS article_id,
                ha.slug         AS slug,
                ha.title        AS title,
                ha.category     AS category,
                ha.locale       AS locale,
                he.source_text  AS source_text,
                1 - (he.embedding <=> CAST(:vec AS vector)) AS sim
            FROM help_embeddings he
            JOIN help_articles ha ON ha.id = he.article_id
            WHERE he.model = :model
              AND ha.is_published = TRUE
              AND ha.locale = :locale
            ORDER BY he.embedding <=> CAST(:vec AS vector)
            LIMIT :lim
            """
        )
        try:
            rows = (await self.db.execute(sql, {
                "vec": str(query_embedding),
                "model": self.model,
                "lim": limit,
                "locale": locale,
            })).all()
        except Exception as exc:
            logger.warning(
                "rag_retrieve_help_failed", error=str(exc)[:200]
            )
            return []

        out: list[RetrievedChunk] = []
        for r in rows:
            if r.sim is None:
                continue
            text_body = _truncate(
                r.source_text or f"{r.title}",
                RAG.MAX_CHUNK_CHARS,
            )
            out.append(RetrievedChunk(
                chunk_id=f"help:{r.slug}",
                kind="help",
                text=text_body,
                cosine=float(r.sim),
                category=None,
                article_id=r.article_id,
                meta={
                    "title": r.title,
                    "slug": r.slug,
                    "help_category": r.category,
                    "locale": r.locale,
                },
            ))
        return out

    # ═══════════════════════════════════════════════════
    # Re-rank + dedupe (unchanged)
    # ═══════════════════════════════════════════════════
    def _rerank(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        now = datetime.now(timezone.utc)
        for c in chunks:
            recency = self._recency_factor(c, now=now)
            sellable = self._sellable_factor(c)
            c.rerank_score = (
                RAG.RERANK_WEIGHT_COSINE * c.cosine
                + RAG.RERANK_WEIGHT_RECENCY * recency
                + RAG.RERANK_WEIGHT_SELLABLE * sellable
            )
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return chunks

    @staticmethod
    def _recency_factor(c: RetrievedChunk, *, now: datetime) -> float:
        if c.kind != "product":
            return 0.5
        ts = c.meta.get("last_updated_at")
        if ts is None:
            return 0.5
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (now - ts).total_seconds() / 86400.0
        if age_days <= 0:
            return 1.0
        if age_days >= 30:
            return 0.0
        return 1.0 - (age_days / 30.0)

    @staticmethod
    def _sellable_factor(c: RetrievedChunk) -> float:
        if c.kind != "product":
            return 0.5
        avail = (c.meta.get("availability") or "").strip()
        if avail == "InStock":
            return 1.0
        return 0.5

    @staticmethod
    def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen_product_ids: set[UUID] = set()
        seen_canonical_ids: set[UUID] = set()
        seen_help_slugs: set[str] = set()
        product_names: list[str] = []
        for c in chunks:
            if c.kind == "product":
                name = (c.meta.get("name") or "").lower().strip()
                if name:
                    product_names.append(name)

        out: list[RetrievedChunk] = []
        for c in chunks:
            if c.kind == "product":
                if c.product_id in seen_product_ids:
                    continue
                seen_product_ids.add(c.product_id)
                out.append(c)
            elif c.kind == "canonical":
                if c.canonical_id in seen_canonical_ids:
                    continue
                cname = (c.meta.get("canonical_name") or "").lower().strip()
                if cname and _name_overlaps(cname, product_names):
                    continue
                seen_canonical_ids.add(c.canonical_id)
                out.append(c)
            elif c.kind == "help":
                slug = c.meta.get("slug") or c.chunk_id
                if slug in seen_help_slugs:
                    continue
                seen_help_slugs.add(slug)
                out.append(c)
        return out

    @staticmethod
    def _pick_category_hint(chunks: list[RetrievedChunk]) -> Optional[str]:
        for c in chunks:
            if c.category:
                return c.category
        return None


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════
def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _truncate(s: str, limit: int) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= limit:
        return s
    cut = s.rfind(" ", 0, limit)
    if cut < limit // 2:
        cut = limit
    return s[:cut].rstrip() + "…"


def _product_text(name, brand, category, price) -> str:
    parts = [
        (name or "").strip(),
        (brand or "").strip(),
        (category or "").strip(),
    ]
    p = _to_float(price)
    if p is not None:
        parts.append(f"{p:.2f} DT")
    return " · ".join(x for x in parts if x)


def _canonical_text(name, site_count, cheapest_site, cheapest_price) -> str:
    n = (name or "").strip()
    sc = int(site_count or 0)
    cp = _to_float(cheapest_price)
    cs = (cheapest_site or "").strip()
    if sc and cs and cp is not None:
        return f"{n} — available at {sc} sites, cheapest at {cs} for {cp:.2f} DT"
    if sc:
        return f"{n} — available at {sc} sites"
    return n


def _name_overlaps(cname: str, product_names: list[str]) -> bool:
    if not cname:
        return False
    c_words = cname.split()
    if not c_words:
        return False
    c_head = set(c_words[:3])
    for pn in product_names:
        p_words = pn.split()
        if not p_words:
            continue
        p_head = set(p_words[:3])
        shared = c_head & p_head
        target = min(3, len(c_head), len(p_head))
        if target and len(shared) >= target:
            return True
    return False