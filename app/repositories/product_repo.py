"""Product repository."""
from typing import Optional, Sequence

from sqlalchemy import and_, asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product import Product
from app.repositories.base import BaseRepository


class ProductRepository(BaseRepository[Product]):
    SORT_FIELDS = {
        "name",
        "price",
        "brand",
        "category",
        "last_updated_at",
        "last_checked",
    }

    def __init__(self, db: AsyncSession):
        super().__init__(db, Product)

    async def paginate(
        self,
        *,
        page: int,
        page_size: int,
        sort_by: str = "last_updated_at",
        sort_order: str = "desc",
        site: Optional[str] = None,
        availability: Optional[str] = None,
        brand: Optional[str] = None,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> tuple[Sequence[Product], int]:
        """
        Paginated, filtered list of products.
        Returns (rows, total).
        """
        conditions = [Product.is_enabled.is_(True)]

        if site:
            conditions.append(Product.site == site.lower())
        if availability:
            conditions.append(Product.availability == availability)
        if brand:
            conditions.append(Product.brand == brand)
        if category:
            conditions.append(Product.category == category)
        if min_price is not None:
            conditions.append(Product.price >= min_price)
        if max_price is not None:
            conditions.append(Product.price <= max_price)

        total = (
            await self.db.execute(
                select(func.count()).select_from(Product).where(and_(*conditions))
            )
        ).scalar_one()

        column = getattr(
            Product,
            sort_by if sort_by in self.SORT_FIELDS else "last_updated_at",
        )
        order = desc(column) if sort_order == "desc" else asc(column)

        rows = (
            await self.db.execute(
                select(Product)
                .where(and_(*conditions))
                .order_by(order)
                .limit(page_size)
                .offset((page - 1) * page_size)
            )
        ).scalars().all()

        return rows, total

    async def get_by_site_and_main_id(
        self, site: str, main_product_id: str
    ) -> Optional[Product]:
        """Fetch a product by its unique (site, main_product_id) pair."""
        result = await self.db.execute(
            select(Product).where(
                Product.site == site.lower(),
                Product.main_product_id == main_product_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_url(self, url: str) -> Optional[Product]:
        result = await self.db.execute(
            select(Product).where(Product.url == url)
        )
        return result.scalar_one_or_none()