"""Recommendation endpoints."""
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.rate_limit import rate_limit_anon, rate_limit_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.recommendation import (
    EventsBatchIn,
    EventsBatchOut,
    RecommendationsResponse,
    SimilarProductsResponse,
)
from app.services.event_service import EventService
from app.services.recommendation_service import RecommendationService

router = APIRouter()


# ═══════════════════════════════════════════════════════
# Events
# ═══════════════════════════════════════════════════════
@router.post(
    "/events",
    response_model=EventsBatchOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[rate_limit_user(times=120, seconds=60)],
)
async def log_events(
    payload: EventsBatchIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> EventsBatchOut:
    svc = EventService(db)
    accepted, rejected, errors = await svc.ingest(
        user_id=user.id, payload=payload,
    )
    return EventsBatchOut(
        accepted=accepted, rejected=rejected, errors=errors,
    )


# ═══════════════════════════════════════════════════════
# Personalized feed
# ═══════════════════════════════════════════════════════
@router.get(
    "/for-me",
    response_model=RecommendationsResponse,
    dependencies=[rate_limit_user(times=60, seconds=60)],
)
async def for_me(
    limit: int = Query(12, ge=1, le=30),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RecommendationsResponse:
    svc = RecommendationService(db)
    return await svc.for_user(user_id=user.id, limit=limit)


# ═══════════════════════════════════════════════════════
# Similar products (works for anonymous too)
# ═══════════════════════════════════════════════════════
@router.get(
    "/similar/{product_id}",
    response_model=SimilarProductsResponse,
    dependencies=[rate_limit_anon(times=120, seconds=60)],
)
async def similar_products(
    product_id: UUID,
    limit: int = Query(8, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
) -> SimilarProductsResponse:
    svc = RecommendationService(db)
    return await svc.similar_to_product(
        product_id=product_id, limit=limit,
    )