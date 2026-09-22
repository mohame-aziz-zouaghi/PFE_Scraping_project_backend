"""Monitoring endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin, get_current_user
from app.core.exceptions import AppException, ServiceUnavailableError
from app.core.logging_config import get_logger
from app.core.monitoring_adapter import monitoring_status
from app.core.rate_limit import rate_limit_user
from app.db.session import get_db
from app.models.product import MonitoringBatch
from app.models.user import User
from app.tasks.monitor_tasks import run_site_monitoring

logger = get_logger(__name__)
router = APIRouter()


@router.get("/batches",
                dependencies=[rate_limit_user(times=100, seconds=60)])
async def list_batches(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    site: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Paginated list of monitoring batches."""
    conditions = []
    if site:
        conditions.append(MonitoringBatch.site == site.lower())
    if status:
        conditions.append(MonitoringBatch.status == status)

    count_stmt = select(func.count()).select_from(MonitoringBatch)
    stmt = select(MonitoringBatch).order_by(desc(MonitoringBatch.start_time))
    if conditions:
        count_stmt = count_stmt.where(*conditions)
        stmt = stmt.where(*conditions)

    total = (await db.execute(count_stmt)).scalar_one()
    rows = (
        await db.execute(stmt.limit(page_size).offset((page - 1) * page_size))
    ).scalars().all()

    return {
        "items": [
            {
                "id": str(r.id),
                "site": r.site,
                "batch_type": r.batch_type,
                "status": r.status,
                "total_products": r.total_products,
                "processed_products": r.processed_products,
                "changed_products": r.changed_products,
                "error_products": r.error_products,
                "start_time": r.start_time.isoformat() if r.start_time else None,
                "end_time": r.end_time.isoformat() if r.end_time else None,
                "duration_seconds": r.duration_seconds,
            }
            for r in rows
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.post("/sites/{site}/trigger",
                 dependencies=[rate_limit_user(times=10, seconds=60)])
async def trigger_site(
    request: Request,
    site: str,
    user: User = Depends(get_current_admin),
):
    """Manually trigger monitoring for a site (admin only)."""
    state = monitoring_status()
    if not state["available"]:
        raise ServiceUnavailableError(
            state.get("reason", "Monitoring engine is not available"),
            details={"monitoring": state},
            retry_after=60,
        )

    site_lower = site.lower()
    task = run_site_monitoring.delay(site_lower)

    logger.info(
        "monitoring_triggered",
        site=site_lower,
        task_id=task.id,
        triggered_by=str(user.id),
    )
    return {"message": "Triggered", "task_id": task.id, "site": site_lower}


@router.get("/status")
async def status(user: User = Depends(get_current_user)):
    """Current monitoring availability."""
    return monitoring_status()