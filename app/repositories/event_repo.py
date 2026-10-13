# app/repositories/event_repo.py
"""Read/write for user_events."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product import Product
from app.models.user_events import EVENT_WEIGHTS, UserEvent


class EventRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ═══════════════════════════════════════════════════════
    # Writes
    # ═══════════════════════════════════════════════════════
    async def insert_batch(
        self,
        *,
        user_id: UUID,
        rows: list[dict],
    ) -> list[UserEvent]:
        """Bulk insert events. `rows` must already be validated and
        carry `event_type`, `product_id` (may be None for search),
        `metadata`, `session_id`."""
        if not rows:
            return []

        # Resolve product_id → canonical_id, base_group_id in one
        # round trip. Search events have no product_id and are
        # skipped here.
        product_ids = {
            r["product_id"] for r in rows
            if r.get("product_id")
        }
        product_map: dict[UUID, tuple[Optional[UUID], Optional[UUID]]] = {}
        if product_ids:
            result = await self.db.execute(
                select(
                    Product.id,
                    Product.canonical_id,
                    Product.base_group_id
                    if hasattr(Product, "base_group_id") else None,
                ).where(Product.id.in_(product_ids))
            )
            for row in result.all():
                product_id = row[0]
                canonical_id = row[1]
                base_group_id = row[2] if len(row) > 2 else None
                product_map[product_id] = (canonical_id, base_group_id)

        objects: list[UserEvent] = []
        for r in rows:
            product_id = r.get("product_id")
            canonical_id, base_group_id = product_map.get(
                product_id, (None, None)
            )
            weight = float(
                r.get("weight")
                or EVENT_WEIGHTS.get(r["event_type"], 1.0)
            )
            objects.append(
                UserEvent(
                    user_id=user_id,
                    event_type=r["event_type"],
                    product_id=product_id,
                    canonical_id=canonical_id,
                    base_group_id=base_group_id,
                    metadata_=r.get("metadata") or {},
                    weight=weight,
                    session_id=r.get("session_id"),
                )
            )

        self.db.add_all(objects)
        await self.db.flush()
        return objects

    # ═══════════════════════════════════════════════════════
    # Reads
    # ═══════════════════════════════════════════════════════
    async def recent_for_user(
        self,
        *,
        user_id: UUID,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> list[UserEvent]:
        stmt = select(UserEvent).where(UserEvent.user_id == user_id)
        if since is not None:
            stmt = stmt.where(UserEvent.created_at >= since)
        stmt = stmt.order_by(UserEvent.created_at.desc()).limit(limit)
        return list((await self.db.execute(stmt)).scalars().all())

    async def count_for_user(self, user_id: UUID) -> int:
        stmt = select(func.count()).select_from(UserEvent).where(
            UserEvent.user_id == user_id
        )
        return int((await self.db.execute(stmt)).scalar_one())

    # ═══════════════════════════════════════════════════════
    # Maintenance
    # ═══════════════════════════════════════════════════════
    async def prune_older_than(self, *, days: int = 180) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self.db.execute(
            delete(UserEvent).where(UserEvent.created_at < cutoff)
        )
        return result.rowcount or 0