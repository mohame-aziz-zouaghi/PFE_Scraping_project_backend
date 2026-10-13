"""Wishlist business logic. Logs user_events on every state change."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.product import Product
from app.models.user_events import EVENT_WEIGHTS
from app.repositories.event_repo import EventRepo
from app.repositories.wishlist_repo import WishlistRepo
from app.schemas.wishlist import WishlistItemOut, WishlistOut


class WishlistService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = WishlistRepo(db)
        self.events = EventRepo(db)

    # ═══════════════════════════════════════════════════════
    # Reads
    # ═══════════════════════════════════════════════════════
    async def get(self, user_id: UUID) -> WishlistOut:
        rows = await self.repo.list_with_products(user_id)
        items = [
            WishlistItemOut(
                product_id=item.product_id,
                added_at=item.created_at,
                name=p.name,
                brand=p.brand,
                category=p.category,
                price=float(p.price) if p.price is not None else None,
                discount_price=(
                    float(p.discount_price)
                    if p.discount_price is not None else None
                ),
                main_image=p.main_image,
                url=p.url,
                site=p.site,
                availability=p.availability,
            )
            for item, p in rows
        ]
        return WishlistOut(count=len(items), items=items)

    async def get_ids(self, user_id: UUID) -> list[UUID]:
        return await self.repo.list_ids(user_id)

    async def is_wishlisted(
        self, *, user_id: UUID, product_id: UUID
    ) -> bool:
        return await self.repo.contains(
            user_id=user_id, product_id=product_id
        )

    # ═══════════════════════════════════════════════════════
    # Toggle — the only mutation the frontend uses
    # ═══════════════════════════════════════════════════════
    async def toggle(
        self, *, user_id: UUID, product_id: UUID
    ) -> bool:
        """Returns True if the item is now wishlisted, False if removed."""
        product = await self.db.get(Product, product_id)
        if product is None:
            raise NotFoundError("Product not found")

        # ── Try to remove first
        removed = await self.repo.remove(
            user_id=user_id, product_id=product_id
        )
        if removed:
            await self.db.commit()
            return False

        # ── Otherwise add
        await self.repo.add(user_id=user_id, product_id=product_id)

        # ── Fire the signal event server-side. We use the
        #    "add_to_cart" event type internally because in this
        #    aggregator the heart *is* the purchase-intent signal —
        #    a user doesn't check out on our site, they heart the
        #    product and later click through to buy.
        #    EVENT_WEIGHTS["add_to_cart"] = 8.0 (stronger than view).
        await self.events.insert_batch(
            user_id=user_id,
            rows=[{
                "event_type": "add_to_cart",
                "product_id": product_id,
                "metadata": {"source": "wishlist"},
                "weight": EVENT_WEIGHTS["add_to_cart"],
            }],
        )
        await self.db.commit()
        return True