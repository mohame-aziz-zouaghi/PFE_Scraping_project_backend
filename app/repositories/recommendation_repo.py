# app/repositories/recommendation_repo.py
"""Materialized recommendation persistence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.recommendation_config import MATERIALIZATION
from app.models.product import Product
from app.models.recommendation import Recommendation


class RecommendationRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_fresh_for_user(
        self,
        user_id: UUID,
        *,
        limit: int = MATERIALIZATION.PER_USER,
    ) -> list[Recommendation]:
        now = datetime.now(timezone.utc)
        stmt = (
            select(Recommendation)
            .where(
                Recommendation.user_id == user_id,
                Recommendation.expires_at > now,
            )
            .order_by(Recommendation.rank.asc())
            .limit(limit)
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def save_batch(
        self,
        *,
        user_id: UUID,
        rows: list[dict],
    ) -> int:
        """Replace the user's existing set with a fresh one."""
        if not rows:
            return 0

        await self.db.execute(
            delete(Recommendation).where(
                Recommendation.user_id == user_id
            )
        )

        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=MATERIALIZATION.TTL_HOURS)

        objects = [
            Recommendation(
                user_id=user_id,
                product_id=r["product_id"],
                rank=r["rank"],
                stage2_score=r["stage2_score"],
                stage3_score=r.get("stage3_score"),
                source=r["source"],
                reason=r.get("reason"),
                generated_at=now,
                expires_at=expires,
            )
            for r in rows
        ]
        self.db.add_all(objects)
        await self.db.flush()
        return len(objects)

    async def purge_expired(self) -> int:
        now = datetime.now(timezone.utc)
        result = await self.db.execute(
            delete(Recommendation).where(
                Recommendation.expires_at <= now
            )
        )
        return result.rowcount or 0

    async def products_by_ids(
        self, ids: list[UUID]
    ) -> dict[UUID, Product]:
        if not ids:
            return {}
        stmt = select(Product).where(Product.id.in_(ids))
        rows = (await self.db.execute(stmt)).scalars().all()
        return {p.id: p for p in rows}