"""Read-side queries for llm_usage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.llm_usage import LLMUsage


class LLMUsageRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def summary_by_endpoint(self, *, hours: int = 24) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(
                LLMUsage.endpoint,
                func.count().label("calls"),
                func.sum(LLMUsage.cost_cents).label("cost_cents"),
                func.avg(LLMUsage.latency_ms).label("avg_ms"),
            )
            .where(LLMUsage.created_at >= since)
            .group_by(LLMUsage.endpoint)
            .order_by(func.sum(LLMUsage.cost_cents).desc())
        )
        rows = (await self.db.execute(stmt)).all()
        return [
            {
                "endpoint": r.endpoint,
                "calls": int(r.calls or 0),
                "cost_usd": float(r.cost_cents or 0) / 100.0,
                "avg_latency_ms": float(r.avg_ms or 0),
            }
            for r in rows
        ]

    async def summary_by_user(
        self, *, hours: int = 24, limit: int = 20
    ) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(
                LLMUsage.user_id,
                func.count().label("calls"),
                func.sum(LLMUsage.cost_cents).label("cost_cents"),
            )
            .where(LLMUsage.created_at >= since, LLMUsage.user_id.isnot(None))
            .group_by(LLMUsage.user_id)
            .order_by(func.sum(LLMUsage.cost_cents).desc())
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).all()
        return [
            {
                "user_id": str(r.user_id),
                "calls": int(r.calls or 0),
                "cost_usd": float(r.cost_cents or 0) / 100.0,
            }
            for r in rows
        ]

    async def cache_hit_ratio(self, *, hours: int = 24) -> float:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        total = (
            await self.db.execute(
                select(func.count())
                .select_from(LLMUsage)
                .where(LLMUsage.created_at >= since)
            )
        ).scalar_one() or 0
        hits = (
            await self.db.execute(
                select(func.count())
                .select_from(LLMUsage)
                .where(
                    LLMUsage.created_at >= since,
                    LLMUsage.cache_hit.is_(True),
                )
            )
        ).scalar_one() or 0
        return (hits / total) if total else 0.0
