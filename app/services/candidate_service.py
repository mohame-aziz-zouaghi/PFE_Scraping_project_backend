"""Stage 1 — candidate generation (search-aware, priority-aware)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.core.recommendation_config import (
    CANDIDATE,
    PROFILE,
    SELLABLE_AVAILABILITY,
    TIER,
)
from app.models.product import Product
from app.models.user_events import UserEvent
from app.models.user_profile import UserProfile

logger = get_logger(__name__)


class CandidateService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model = settings.active_embedding_model

    async def generate(
        self,
        *,
        user_id: UUID,
        profile: Optional[UserProfile],
        limit: int = CANDIDATE.MAX_CANDIDATES,
        exclude: Optional[set[UUID]] = None,
        include: Optional[list[UUID]] = None,
    ) -> list[UUID]:
        """
        Build the candidate pool.

        Order of precedence:
          0. `include` — explicit priority-band products (fresh
             wishlist / cart / search). Always first.
          A′. Literal matches for the user's recent search queries.
          A.  Content neighbours of recent views.
          B.  Best-sellers in the user's top categories.
          C.  Newest enabled products (fallback).
        """
        exclude = exclude or set()
        include = include or []
        seen: set[UUID] = set(exclude)
        candidates: list[UUID] = []

        def add_batch(ids: list[UUID]) -> None:
            for pid in ids:
                if pid in seen:
                    continue
                seen.add(pid)
                candidates.append(pid)
                if len(candidates) >= limit:
                    return

        # ── Source 0: explicitly-included priority-band products.
        #    These are the fresh wishlist / cart / search hits the
        #    caller wants back. They bypass every other source.
        add_batch(include)

        # ── Source A′: literal matches for recent search queries.
        #    Highest-priority organic source — this is what makes
        #    "maison" searches surface housing items immediately.
        if len(candidates) < limit:
            search_hits = await self._matches_recent_searches(
                user_id=user_id
            )
            add_batch(search_hits)

        # ── Source A: content neighbours of the user's recent views
        #    (bounded by the same window the profile uses).
        if len(candidates) < limit:
            seeds = await self._recent_viewed_products(
                user_id=user_id, limit=CANDIDATE.RECENT_VIEWS_SEED
            )
            for pid in seeds:
                neighbours = await self._content_neighbours(
                    product_id=pid,
                    limit=CANDIDATE.NEIGHBOURS_PER_SEED,
                )
                add_batch(neighbours)
                if len(candidates) >= limit:
                    break

        # ── Source B: best-sellers in the user's top categories.
        #    Fresh-tier categories first so today's interest wins.
        if len(candidates) < limit and profile and profile.preferred_categories:
            ordered = sorted(
                profile.preferred_categories,
                key=lambda c: (
                    # Fresh / mixed first, then long
                    c.get("tier") not in ("fresh", "mixed")
                    if isinstance(c, dict) else True,
                    # Then by descending weight
                    -float(
                        (c.get("weight") if isinstance(c, dict) else 0)
                        or 0
                    ),
                ),
            )
            for cat in ordered[: CANDIDATE.TOP_CATEGORIES]:
                name = (
                    cat.get("name") if isinstance(cat, dict) else None
                )
                if not name:
                    continue
                cohort = await self._category_cohort(
                    category=name,
                    limit=CANDIDATE.CATEGORY_COHORT_SIZE,
                )
                add_batch(cohort)
                if len(candidates) >= limit:
                    break

        # ── Source C: newest enabled as a fallback.
        if len(candidates) < limit:
            fallback = await self._newest_products(
                limit=limit - len(candidates)
            )
            add_batch(fallback)

        return candidates[:limit]

    # ═══════════════════════════════════════════════════════
    # Search-driven candidates
    # ═══════════════════════════════════════════════════════
    async def _matches_recent_searches(
        self, *, user_id: UUID
    ) -> list[UUID]:
        """
        Find products whose name/category matches any of the user's
        search queries from the last TIER.TODAY_HOURS. Freshest
        query wins.
        """
        since = datetime.now(timezone.utc) - timedelta(
            hours=TIER.TODAY_HOURS
        )

        rows = (await self.db.execute(
            select(UserEvent.created_at, UserEvent.metadata_)
            .where(
                UserEvent.user_id == user_id,
                UserEvent.event_type == "search",
                UserEvent.created_at >= since,
            )
            .order_by(UserEvent.created_at.desc())
            .limit(20)
        )).all()

        # Dedupe queries, freshest first.
        seen_q: list[str] = []
        for _, meta in rows:
            q = (meta or {}).get("query")
            if q and q not in seen_q:
                seen_q.append(q)
        if not seen_q:
            return []

        out: list[UUID] = []
        for q in seen_q[: CANDIDATE.RECENT_SEARCHES_TO_MATCH]:
            stmt = (
                select(Product.id)
                .where(
                    Product.is_enabled.is_(True),
                    Product.availability.in_(SELLABLE_AVAILABILITY),
                    or_(
                        Product.name.ilike(f"%{q}%"),
                        Product.category.ilike(f"%{q}%"),
                    ),
                )
                .order_by(
                    Product.price_change_count.desc().nullslast(),
                    Product.last_updated_at.desc(),
                )
                .limit(CANDIDATE.SEARCH_COHORT_SIZE)
            )
            for (pid,) in (await self.db.execute(stmt)).all():
                if pid not in out:
                    out.append(pid)
        return out

    # ═══════════════════════════════════════════════════════
    # Internals
    # ═══════════════════════════════════════════════════════
    async def _recent_viewed_products(
        self, *, user_id: UUID, limit: int
    ) -> list[UUID]:
        """
        Most recent distinct products the user interacted with,
        restricted to the profile event window.
        """
        since = datetime.now(timezone.utc) - timedelta(
            days=PROFILE.EVENT_WINDOW_DAYS
        )
        stmt = (
            select(UserEvent.product_id)
            .where(
                UserEvent.user_id == user_id,
                UserEvent.product_id.isnot(None),
                UserEvent.created_at >= since,
            )
            .order_by(UserEvent.created_at.desc())
            .limit(limit * 4)
        )
        rows = (await self.db.execute(stmt)).all()
        out: list[UUID] = []
        seen: set[UUID] = set()
        for (pid,) in rows:
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            out.append(pid)
            if len(out) >= limit:
                break
        return out

    async def _content_neighbours(
        self, *, product_id: UUID, limit: int
    ) -> list[UUID]:
        sql = text(
            """
            SELECT pe2.product_id
            FROM product_embeddings pe1
            JOIN product_embeddings pe2
              ON pe2.model = pe1.model
             AND pe2.product_id <> pe1.product_id
            JOIN products p ON p.id = pe2.product_id
            WHERE pe1.product_id = :pid
              AND pe1.model = :model
              AND p.is_enabled = TRUE
              AND p.availability = ANY(:sellable)
            ORDER BY pe1.embedding <=> pe2.embedding
            LIMIT :limit
            """
        )
        try:
            rows = (await self.db.execute(sql, {
                "pid": str(product_id),
                "model": self.model,
                "limit": limit,
                "sellable": list(SELLABLE_AVAILABILITY),
            })).all()
        except Exception as exc:
            logger.warning(
                "content_neighbours_failed",
                product_id=str(product_id),
                error=str(exc)[:200],
            )
            return []
        return [r[0] for r in rows]

    async def _category_cohort(
        self, *, category: str, limit: int
    ) -> list[UUID]:
        stmt = (
            select(Product.id)
            .where(
                Product.is_enabled.is_(True),
                Product.category == category,
                Product.availability.in_(SELLABLE_AVAILABILITY),
            )
            .order_by(
                Product.price_change_count.desc().nullslast(),
                Product.last_updated_at.desc(),
            )
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).all()
        return [r[0] for r in rows]

    async def _newest_products(self, *, limit: int) -> list[UUID]:
        stmt = (
            select(Product.id)
            .where(
                Product.is_enabled.is_(True),
                Product.availability.in_(SELLABLE_AVAILABILITY),
            )
            .order_by(Product.last_updated_at.desc())
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).all()
        return [r[0] for r in rows]