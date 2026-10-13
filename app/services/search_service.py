"""
Search service — text, suggestions, semantic.

Design:
  - Text search uses Postgres ILIKE with a pg_trgm GIN index.
  - Suggestions fan-out is one UNION ALL query instead of three.
  - Every search is logged as a user_event with weight 5 so the
    recommendation engine can use search intent as a signal.
  - Duplicate searches within 60 seconds are dropped (protects
    against client refetch storms).
  - Result caching is handled at the API layer via Redis.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    and_,
    desc,
    func,
    literal,
    or_,
    select,
    text,
    union_all,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.models.product import Product
from app.schemas.search import (
    SearchResponse,
    SearchResult,
    SemanticSearchRequest,
    SuggestionsResponse,
    SuggestionItem,
    TextSearchRequest,
)

logger = get_logger(__name__)


# Search event weight — higher than view (1), lower than wishlist (5).
SEARCH_EVENT_WEIGHT = 5.0

# Duplicate-search dedup window.
SEARCH_DEDUP_SECONDS = 60


class SearchService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ═══════════════════════════════════════════════════════
    # TEXT SEARCH
    # ═══════════════════════════════════════════════════════
    async def text_search(
        self,
        payload: TextSearchRequest,
        *,
        user_id: Optional[UUID] = None,
    ) -> SearchResponse:
        start = time.perf_counter()

        q = payload.query.strip()
        if not q:
            return _empty_response(
                payload.query, payload.page, payload.page_size
            )

        like_pattern = f"%{q}%"

        conditions = _base_conditions(payload)
        conditions.append(
            or_(
                Product.name.ilike(like_pattern),
                Product.brand.ilike(like_pattern),
                Product.category.ilike(like_pattern),
            )
        )

        # ── Count
        total = (
            await self.db.execute(
                select(func.count())
                .select_from(Product)
                .where(and_(*conditions))
            )
        ).scalar_one()

        # ── Order
        order = _ordering(payload.sort_by)

        # ── Page
        stmt = (
            select(Product)
            .where(and_(*conditions))
            .order_by(order)
            .limit(payload.page_size)
            .offset((payload.page - 1) * payload.page_size)
        )
        rows = (await self.db.execute(stmt)).scalars().all()

        items = [self._to_result(r, score=1.0) for r in rows]

        # ── Log signal (logged-in users only, deduped server-side)
        if user_id is not None:
            await self._log_search_event(
                user_id=user_id,
                query=q,
                result_count=total,
                session_id=payload.session_id,
            )

        return SearchResponse(
            query=q,
            items=items,
            total=total,
            page=payload.page,
            page_size=payload.page_size,
            total_pages=(total + payload.page_size - 1) // payload.page_size,
            took_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    # ═══════════════════════════════════════════════════════
    # SUGGESTIONS
    # ═══════════════════════════════════════════════════════
    async def suggestions(
        self, query: str, limit: int
    ) -> SuggestionsResponse:
        q = query.strip()
        if not q:
            return SuggestionsResponse(query="", items=[])
        like_pattern = f"%{q}%"

        # One UNION ALL query instead of three sequential scans.
        name_q = (
            select(
                Product.name.label("text"),
                literal("product").label("kind"),
                func.count().label("cnt"),
            )
            .where(Product.is_enabled.is_(True))
            .where(Product.name.ilike(like_pattern))
            .where(Product.name.isnot(None))
            .group_by(Product.name)
        )
        brand_q = (
            select(
                Product.brand.label("text"),
                literal("brand").label("kind"),
                func.count().label("cnt"),
            )
            .where(Product.is_enabled.is_(True))
            .where(Product.brand.ilike(like_pattern))
            .where(Product.brand.isnot(None))
            .group_by(Product.brand)
        )
        cat_q = (
            select(
                Product.category.label("text"),
                literal("category").label("kind"),
                func.count().label("cnt"),
            )
            .where(Product.is_enabled.is_(True))
            .where(Product.category.ilike(like_pattern))
            .where(Product.category.isnot(None))
            .group_by(Product.category)
        )

        combined = union_all(name_q, brand_q, cat_q).subquery()
        stmt = (
            select(combined)
            .order_by(desc(combined.c.cnt))
            .limit(limit * 3)
        )
        rows = (await self.db.execute(stmt)).all()

        # Dedup by text across kind.
        seen: set[str] = set()
        items: list[SuggestionItem] = []
        for row in rows:
            key = (row.text or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(
                SuggestionItem(
                    text=row.text,
                    type=row.kind,
                    count=int(row.cnt),
                )
            )
            if len(items) >= limit:
                break

        return SuggestionsResponse(query=q, items=items)

    # ═══════════════════════════════════════════════════════
    # SEMANTIC SEARCH
    # ═══════════════════════════════════════════════════════
    async def semantic_search(
        self,
        payload: SemanticSearchRequest,
        *,
        user_id: Optional[UUID] = None,
    ) -> SearchResponse:
        start = time.perf_counter()
        q = payload.query.strip()
        if not q:
            return _empty_response(payload.query, 1, payload.limit)

        # ── 1. Embed the query
        embedding_vector: Optional[list[float]] = None
        try:
            from app.core.embeddings import embed_text

            embedding_vector = await embed_text(q)
        except Exception as exc:
            logger.warning(
                "semantic_embed_failed", error=str(exc)[:200]
            )

        # ── 2. Fallback to text search if embeddings unavailable
        if embedding_vector is None:
            fallback = await self.text_search(
                TextSearchRequest(
                    query=q,
                    site=payload.site,
                    page=1,
                    page_size=payload.limit,
                ),
                user_id=user_id,
            )
            fallback.took_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )
            return fallback

        # ── 3. Vector search over product_embeddings
        from app.config import settings as _settings

        sql = text(
            """
            SELECT p.*, 1 - (pe.embedding <=> CAST(:vec AS vector)) AS sim
            FROM product_embeddings pe
            JOIN products p ON p.id = pe.product_id
            WHERE pe.model = :model
              AND p.is_enabled = TRUE
              AND (:site IS NULL OR p.site = :site)
              AND (1 - (pe.embedding <=> CAST(:vec AS vector))) >= :min_sim
            ORDER BY pe.embedding <=> CAST(:vec AS vector)
            LIMIT :lim
            """
        )
        rows = (
            await self.db.execute(
                sql,
                {
                    "vec": str(embedding_vector),
                    "model": _settings.active_embedding_model,
                    "site": payload.site.lower() if payload.site else None,
                    "min_sim": payload.min_similarity,
                    "lim": payload.limit,
                },
            )
        ).all()

        items: list[SearchResult] = []
        for r in rows:
            items.append(
                SearchResult(
                    product_id=r.id,
                    site=r.site,
                    name=r.name or "",
                    brand=r.brand,
                    category=r.category,
                    price=float(r.price or 0),
                    availability=r.availability or "unknown",
                    main_image=r.main_image,
                    url=r.url,
                    score=float(r.sim or 0),
                )
            )

        if user_id is not None:
            await self._log_search_event(
                user_id=user_id,
                query=q,
                result_count=len(items),
                session_id=payload.session_id,
            )

        return SearchResponse(
            query=q,
            items=items,
            total=len(items),
            page=1,
            page_size=payload.limit,
            total_pages=1,
            took_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    # ═══════════════════════════════════════════════════════
    # SIGNAL LOGGING
    # ═══════════════════════════════════════════════════════
    async def _log_search_event(
        self,
        *,
        user_id: UUID,
        query: str,
        result_count: int,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Record a search event, but skip if the same user searched the
        same normalized query in the last 60 seconds.

        Protects against client-side refetch storms: react-query
        retries, window-focus refetches, and any accidental
        double-fire on mount. A real user never types the same query
        twice within a minute.
        """
        try:
            from app.models.user_events import UserEvent

            query_norm = query.strip().lower()
            if not query_norm:
                return

            cutoff = datetime.now(timezone.utc) - timedelta(
                seconds=SEARCH_DEDUP_SECONDS
            )

            # Dedup check: any search event by this user with the
            # same normalized query in the last N seconds.
            recent_stmt = (
                select(func.count())
                .select_from(UserEvent)
                .where(UserEvent.user_id == user_id)
                .where(UserEvent.event_type == "search")
                .where(UserEvent.created_at >= cutoff)
                .where(
                    func.lower(
                        UserEvent.metadata_["query"].astext
                    )
                    == query_norm
                )
            )
            recent = int(
                (await self.db.execute(recent_stmt)).scalar_one()
            )
            if recent > 0:
                logger.debug(
                    "search_event_deduped",
                    user_id=str(user_id),
                    query=query_norm,
                )
                return

            first_product_id = await self._first_result_product_id(query)
            metadata = {
                "query": query[:200],
                "result_count": int(result_count),
            }
            self.db.add(
                UserEvent(
                    user_id=user_id,
                    event_type="search",
                    product_id=first_product_id,
                    metadata_=metadata,
                    weight=SEARCH_EVENT_WEIGHT,
                    session_id=session_id,
                )
            )
            await self.db.commit()
        except Exception as exc:
            await self.db.rollback()
            logger.warning(
                "search_event_failed", error=str(exc)[:200]
            )

    async def _first_result_product_id(
        self, query: str
    ) -> Optional[UUID]:
        like = f"%{query}%"
        stmt = (
            select(Product.id)
            .where(Product.is_enabled.is_(True))
            .where(Product.name.ilike(like))
            .limit(1)
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    # ═══════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _to_result(p: Product, score: float = 1.0) -> SearchResult:
        return SearchResult(
            product_id=p.id,
            site=p.site,
            name=p.name or "",
            brand=p.brand,
            category=p.category,
            price=float(p.price or 0),
            availability=p.availability or "unknown",
            main_image=p.main_image,
            url=p.url,
            score=score,
        )


# ═══════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════
def _base_conditions(payload: TextSearchRequest) -> list:
    conds = [Product.is_enabled.is_(True)]
    if payload.site:
        conds.append(Product.site == payload.site.lower())
    if payload.brand:
        conds.append(Product.brand == payload.brand)
    if payload.category:
        conds.append(Product.category == payload.category)
    if payload.availability:
        conds.append(Product.availability == payload.availability)
    if payload.min_price is not None:
        conds.append(Product.price >= payload.min_price)
    if payload.max_price is not None:
        conds.append(Product.price <= payload.max_price)
    return conds


def _ordering(sort_by: str):
    if sort_by == "price":
        return Product.price.asc()
    if sort_by == "price_desc":
        return Product.price.desc()
    if sort_by == "recent":
        return desc(Product.last_updated_at)
    # relevance — shortest name wins
    return func.length(Product.name).asc()


def _empty_response(
    query: str, page: int, page_size: int
) -> SearchResponse:
    return SearchResponse(
        query=query,
        items=[],
        total=0,
        page=page,
        page_size=page_size,
        total_pages=1,
        took_ms=0.0,
    )


def text_search_cache_key(payload: TextSearchRequest) -> str:
    from app.core.cache import make_key

    return make_key(
        "search",
        "text",
        payload.query.lower().strip(),
        payload.site or "",
        payload.brand or "",
        payload.category or "",
        payload.availability or "",
        payload.min_price or "",
        payload.max_price or "",
        payload.page,
        payload.page_size,
        payload.sort_by,
    )


def semantic_search_cache_key(payload: SemanticSearchRequest) -> str:
    from app.core.cache import make_key

    return make_key(
        "search",
        "sem",
        payload.query.lower().strip(),
        payload.site or "",
        payload.limit,
        payload.min_similarity,
    )


def suggestions_cache_key(q: str, limit: int) -> str:
    from app.core.cache import make_key

    return make_key("search", "sug", q.lower().strip(), limit)