"""Alerts endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.rate_limit import rate_limit_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.alert import AlertResolveResponse, PaginatedAlerts
from app.services.alert_service import AlertService

router = APIRouter()


@router.get(
    "",
    response_model=PaginatedAlerts,
    dependencies=[rate_limit_user(times=100, seconds=60)],
)
async def list_alerts(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    site: Optional[str] = Query(None, max_length=50),
    alert_type: Optional[str] = Query(None, max_length=50),
    severity: Optional[str] = Query(None, max_length=20),
    is_resolved: Optional[bool] = Query(False),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List alerts with filters + pagination."""
    return await AlertService(db).list_alerts(
        page=page,
        page_size=page_size,
        site=site,
        alert_type=alert_type,
        severity=severity,
        is_resolved=is_resolved,
    )


@router.post(
    "/{alert_id}/resolve",
    response_model=AlertResolveResponse,
    dependencies=[rate_limit_user(times=30, seconds=60)],
)
async def resolve_alert(
    request: Request,
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Mark an alert as resolved."""
    return await AlertService(db).resolve(alert_id)