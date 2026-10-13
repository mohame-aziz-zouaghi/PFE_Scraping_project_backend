"""Wishlist endpoints."""
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.rate_limit import rate_limit_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.wishlist import (
    WishlistIdsOut,
    WishlistOut,
    WishlistToggleIn,
    WishlistToggleOut,
)
from app.services.wishlist_service import WishlistService

router = APIRouter()


@router.get(
    "",
    response_model=WishlistOut,
    dependencies=[rate_limit_user(times=120, seconds=60)],
)
async def get_wishlist(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> WishlistOut:
    """Full wishlist with product snapshots."""
    return await WishlistService(db).get(user.id)


@router.get(
    "/ids",
    response_model=WishlistIdsOut,
    dependencies=[rate_limit_user(times=240, seconds=60)],
)
async def get_wishlist_ids(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> WishlistIdsOut:
    """Just the product IDs. Cheap — the frontend uses this to
    paint heart icons on product cards without fetching the full
    list."""
    ids = await WishlistService(db).get_ids(user.id)
    return WishlistIdsOut(product_ids=ids)


@router.post(
    "/toggle",
    response_model=WishlistToggleOut,
    status_code=status.HTTP_200_OK,
    dependencies=[rate_limit_user(times=120, seconds=60)],
)
async def toggle_wishlist(
    payload: WishlistToggleIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> WishlistToggleOut:
    """Add or remove. Idempotent — callers can't accidentally
    double-add."""
    in_wishlist = await WishlistService(db).toggle(
        user_id=user.id, product_id=payload.product_id,
    )
    return WishlistToggleOut(in_wishlist=in_wishlist)