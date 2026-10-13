"""Wishlist schemas."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class WishlistItemOut(BaseModel):
    """One heart. Snapshot fields avoid a second request for the
    product details."""

    product_id: UUID
    added_at: datetime

    # Product snapshot
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    discount_price: Optional[float] = None
    main_image: Optional[str] = None
    url: Optional[str] = None
    site: Optional[str] = None
    availability: Optional[str] = None


class WishlistOut(BaseModel):
    count: int
    items: list[WishlistItemOut]


class WishlistToggleIn(BaseModel):
    product_id: UUID


class WishlistToggleOut(BaseModel):
    in_wishlist: bool


class WishlistIdsOut(BaseModel):
    """Just the IDs — the frontend uses this to paint heart icons
    on every product card without fetching the full item list."""

    product_ids: list[UUID]