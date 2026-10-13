"""Fetch priority-band product IDs directly from user_events.

Each band is "the products this user acted on with this event
type, within this time window, ordered by recency". The
recommendation service uses these to guarantee that fresh
wishlist / search / cart products always appear at the top of
the feed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product import Product
from app.models.user_events import UserEvent
from app.core.recommendation_config import SELLABLE_AVAILABILITY


class PriorityBandService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def wishlist_band(
        self, *, user_id: UUID, hours: int = 24, limit: int = 10
    ) -> list[UUID]:
        """Products the user wishlisted inside `hours`, newest first."""
        return await self._products_by_event(
            user_id=user_id, event_type="wishlist",
            hours=hours, limit=limit,
        )

    async def cart_band(
        self, *, user_id: UUID, hours: int = 24, limit: int = 10
    ) -> list[UUID]:
        return await self._products_by_event(
            user_id=user_id, event_type="add_to_cart",
            hours=hours, limit=limit,
        )

    async def click_band(
        self, *, user_id: UUID, hours: int = 6, limit: int = 10
    ) -> list[UUID]:
        return await self._products_by_event(
            user_id=user_id, event_type="click",
            hours=hours, limit=limit,
        )

    async def dwell_band(
        self, *, user_id: UUID, hours: int = 6, limit: int = 10
    ) -> list[UUID]:
        return await self._products_by_event(
            user_id=user_id, event_type="dwell",
            hours=hours, limit=limit,
        )

    async def search_band(
        self, *, user_id: UUID, hours: int = 24, limit: int = 10
    ) -> list[UUID]:
        """
        Products that literally match one of the user's recent
        search queries (name/category ILIKE). Searches have no
        product_id, so this is a text-match against the
        metadata.query field.
        """
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        rows = (await self.db.execute(
            select(UserEvent.metadata_)
            .where(
                UserEvent.user_id == user_id,
                UserEvent.event_type == "search",
                UserEvent.created_at >= since,
            )
            .order_by(UserEvent.created_at.desc())
            .limit(10)
        )).all()

        seen_q: list[str] = []
        for (meta,) in rows:
            q = (meta or {}).get("query")
            if q and q not in seen_q:
                seen_q.append(q)
        if not seen_q:
            return []

        out: list[UUID] = []
        for q in seen_q[:3]:
            stmt = (
                select(Product.id)
                .where(
                    Product.is_enabled.is_(True),
                    Product.availability.in_(SELLABLE_AVAILABILITY),
                    (Product.name.ilike(f"%{q}%"))
                    | (Product.category.ilike(f"%{q}%")),
                )
                .order_by(Product.last_updated_at.desc())
                .limit(limit)
            )
            for (pid,) in (await self.db.execute(stmt)).all():
                if pid not in out:
                    out.append(pid)
        return out

    # ────────────────────────────────────────────────
    async def _products_by_event(
        self, *, user_id: UUID, event_type: str,
        hours: int, limit: int,
    ) -> list[UUID]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(UserEvent.product_id)
            .where(
                UserEvent.user_id == user_id,
                UserEvent.event_type == event_type,
                UserEvent.product_id.isnot(None),
                UserEvent.created_at >= since,
            )
            .order_by(UserEvent.created_at.desc())
            .limit(limit * 3)          # over-fetch to dedupe
        )
        rows = (await self.db.execute(stmt)).all()

        # Dedupe, keep freshest-first order.
        out: list[UUID] = []
        for (pid,) in rows:
            if pid is None or pid in out:
                continue
            out.append(pid)
            if len(out) >= limit:
                break
        return out