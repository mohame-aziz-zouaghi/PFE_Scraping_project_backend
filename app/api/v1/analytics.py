"""Analytics endpoints."""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.rate_limit import rate_limit_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.analytics import (
    AlertSummary,
    AvailabilityTimelineResponse,
    PriceDropsResponse,
    PriceTrendResponse,
    SitesOverviewResponse,
    VolatileProductsResponse,
)
from app.services.analytics_service import AnalyticsService

router = APIRouter()


@router.get("/price-trends", response_model=PriceTrendResponse,
    dependencies=[rate_limit_user(times=60, seconds=60)])
async def get_price_trends(
    request: Request,
    site: Optional[str] = Query(None, max_length=50),
    product_id: Optional[UUID] = Query(None),
    period_days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get price trend line for a site or product."""
    service = AnalyticsService(db)
    return await service.get_price_trends(
        site=site,
        product_id=str(product_id) if product_id else None,
        period_days=period_days,
    )


@router.get("/volatile-products", response_model=VolatileProductsResponse,
                dependencies=[rate_limit_user(times=60, seconds=60)])
async def get_volatile_products(
    request: Request,
    site: Optional[str] = Query(None, max_length=50),
    period_days: int = Query(7, ge=1, le=90),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get top-N most volatile products."""
    service = AnalyticsService(db)
    return await service.get_volatile_products(
        site=site, period_days=period_days, limit=limit
    )


@router.get("/price-drops", response_model=PriceDropsResponse,
                dependencies=[rate_limit_user(times=60, seconds=60)])
async def get_price_drops(
    request: Request,
    site: Optional[str] = Query(None, max_length=50),
    period_days: int = Query(7, ge=1, le=90),
    min_change_percent: float = Query(5.0, ge=0, le=100),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get biggest price drops in a period."""
    service = AnalyticsService(db)
    return await service.get_price_drops(
        site=site,
        period_days=period_days,
        min_change_percent=min_change_percent,
        limit=limit,
    )


@router.get("/alerts-summary", response_model=AlertSummary,
                dependencies=[rate_limit_user(times=60, seconds=60)])
async def get_alert_summary(
    request: Request,
    site: Optional[str] = Query(None, max_length=50),
    period_days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get alert counts grouped by site/type/severity."""
    service = AnalyticsService(db)
    return await service.get_alert_summary(site=site, period_days=period_days)


@router.get("/availability-timeline", response_model=AvailabilityTimelineResponse,
                dependencies=[rate_limit_user(times=60, seconds=60)])
async def get_availability_timeline(
    request: Request,
    site: Optional[str] = Query(None, max_length=50),
    period_days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get in-stock / out-of-stock counts over time."""
    service = AnalyticsService(db)
    return await service.get_availability_timeline(
        site=site, period_days=period_days
    )


@router.get("/sites-overview", response_model=SitesOverviewResponse,
                dependencies=[rate_limit_user(times=60, seconds=60)])
async def get_sites_overview(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Aggregate overview per site."""
    service = AnalyticsService(db)
    return await service.get_sites_overview()