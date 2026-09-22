"""Alert repository."""
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product import MonitoringAlert
from app.repositories.base import BaseRepository


class AlertRepository(BaseRepository[MonitoringAlert]):
    def __init__(self, db: AsyncSession):
        super().__init__(db, MonitoringAlert)

    async def paginate(
        self,
        *,
        page: int,
        page_size: int,
        site: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        is_resolved: Optional[bool] = None,
    ) -> tuple[Sequence[MonitoringAlert], int]:
        """
        Paginated, filtered list of alerts.
        Returns (rows, total).
        """
        conditions = []
        if is_resolved is not None:
            conditions.append(MonitoringAlert.is_resolved.is_(is_resolved))
        if site:
            conditions.append(MonitoringAlert.site == site.lower())
        if alert_type:
            conditions.append(MonitoringAlert.alert_type == alert_type)
        if severity:
            conditions.append(MonitoringAlert.severity == severity)

        count_stmt = select(func.count()).select_from(MonitoringAlert)
        stmt = (
            select(MonitoringAlert)
            .order_by(desc(MonitoringAlert.created_at))
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
        if conditions:
            count_stmt = count_stmt.where(and_(*conditions))
            stmt = stmt.where(and_(*conditions))

        total = (await self.db.execute(count_stmt)).scalar_one()
        rows = (await self.db.execute(stmt)).scalars().all()

        return rows, total

    async def resolve(self, alert_id: int) -> Optional[MonitoringAlert]:
        """Mark an alert as resolved. Returns the updated row, or None."""
        result = await self.db.execute(
            select(MonitoringAlert).where(MonitoringAlert.id == alert_id)
        )
        alert = result.scalar_one_or_none()
        if not alert:
            return None

        alert.is_resolved = True
        alert.resolved_at = datetime.now(timezone.utc)
        await self.db.flush()
        return alert

    async def unresolved_count(self, site: Optional[str] = None) -> int:
        conditions = [MonitoringAlert.is_resolved.is_(False)]
        if site:
            conditions.append(MonitoringAlert.site == site.lower())
        result = await self.db.execute(
            select(func.count())
            .select_from(MonitoringAlert)
            .where(and_(*conditions))
        )
        return int(result.scalar_one() or 0)

    async def recent(
        self, limit: int = 20, site: Optional[str] = None
    ) -> Sequence[MonitoringAlert]:
        stmt = (
            select(MonitoringAlert)
            .order_by(desc(MonitoringAlert.created_at))
            .limit(limit)
        )
        if site:
            stmt = stmt.where(MonitoringAlert.site == site.lower())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())