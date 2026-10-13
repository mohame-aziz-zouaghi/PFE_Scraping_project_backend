"""Read/write for wishlist_items."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product import Product
from app.models.wishlist import WishlistItem


class WishlistRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_with_products(
        self, user_id: UUID
    ) -> list[tuple[WishlistItem, Product]]:
        stmt = (
            select(WishlistItem, Product)
            .join(Product, Product.id == WishlistItem.product_id)
            .where(WishlistItem.user_id == user_id)
            .order_by(WishlistItem.created_at.desc())
        )
        return list((await self.db.execute(stmt)).all())

    async def list_ids(self, user_id: UUID) -> list[UUID]:
        stmt = (
            select(WishlistItem.product_id)
            .where(WishlistItem.user_id == user_id)
            .order_by(WishlistItem.created_at.desc())
        )
        rows = (await self.db.execute(stmt)).all()
        return [r[0] for r in rows]

    async def contains(
        self, *, user_id: UUID, product_id: UUID
    ) -> bool:
        stmt = (
            select(func.count())
            .select_from(WishlistItem)
            .where(
                WishlistItem.user_id == user_id,
                WishlistItem.product_id == product_id,
            )
        )
        return int((await self.db.execute(stmt)).scalar_one()) > 0

    async def add(
        self, *, user_id: UUID, product_id: UUID
    ) -> tuple[WishlistItem, bool]:
        """Idempotent. Returns (item, created)."""
        existing = await self.db.execute(
            select(WishlistItem).where(
                WishlistItem.user_id == user_id,
                WishlistItem.product_id == product_id,
            )
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            return row, False
        item = WishlistItem(user_id=user_id, product_id=product_id)
        self.db.add(item)
        await self.db.flush()
        return item, True

    async def remove(
        self, *, user_id: UUID, product_id: UUID
    ) -> bool:
        result = await self.db.execute(
            delete(WishlistItem).where(
                WishlistItem.user_id == user_id,
                WishlistItem.product_id == product_id,
            )
        )
        return (result.rowcount or 0) > 0

    async def count(self, user_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(WishlistItem)
            .where(WishlistItem.user_id == user_id)
        )
        return int((await self.db.execute(stmt)).scalar_one())