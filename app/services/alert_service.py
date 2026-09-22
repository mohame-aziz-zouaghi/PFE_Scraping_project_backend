"""Alert business logic."""
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.repositories.alert_repo import AlertRepository
from app.schemas.alert import AlertResolveResponse, AlertResponse, PaginatedAlerts


class AlertService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repository = AlertRepository(db)

    async def list_alerts(
        self,
        page: int,
        page_size: int,
        site: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        is_resolved: Optional[bool] = False,
    ) -> PaginatedAlerts:
        rows, total = await self.repository.paginate(   # ← was .list()
            page=page,
            page_size=page_size,
            site=site,
            alert_type=alert_type,
            severity=severity,
            is_resolved=is_resolved,
        )
        return PaginatedAlerts(
            items=[AlertResponse.model_validate(row) for row in rows],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
        )

    async def resolve(self, alert_id: int) -> AlertResolveResponse:
        alert = await self.repository.resolve(alert_id)
        if not alert:
            raise NotFoundError(f"Alert {alert_id} not found")
        await self.db.commit()
        await self.db.refresh(alert)
        return AlertResolveResponse(
            message="Alert resolved",
            alert=AlertResponse.model_validate(alert),
        )