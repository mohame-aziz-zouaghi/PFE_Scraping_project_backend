"""Product business logic."""
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.product import PriceHistory, Product, ProductChange
from app.schemas.product import (
    ChangeHistoryItem,
    ChangeHistoryResponse,
    PaginatedProducts,
    PriceHistoryItem,
    PriceHistoryResponse,
    ProductDetail,
    ProductSummary,
)

_SORTABLE = {"name", "price", "brand", "category", "last_updated_at", "last_checked"}


class ProductService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_products(
        self,
        page: int,
        page_size: int,
        site: Optional[str] = None,
        availability: Optional[str] = None,
        brand: Optional[str] = None,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        sort_by: str = "last_updated_at",
        sort_order: str = "desc",
    ) -> PaginatedProducts:
        conditions = [Product.is_enabled == True]
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

        count_stmt = (
            select(func.count()).select_from(Product).where(and_(*conditions))
        )
        total = (await self.db.execute(count_stmt)).scalar_one()

        field = sort_by if sort_by in _SORTABLE else "last_updated_at"
        sort_col = getattr(Product, field)
        order_fn = desc if sort_order == "desc" else asc

        stmt = (
            select(Product)
            .where(and_(*conditions))
            .order_by(order_fn(sort_col))
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
        rows = (await self.db.execute(stmt)).scalars().all()

        return PaginatedProducts(
            items=[ProductSummary.model_validate(r) for r in rows],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
        )

    async def get_product(self, product_id: UUID) -> ProductDetail:
        stmt = select(Product).where(Product.id == product_id)
        product = (await self.db.execute(stmt)).scalar_one_or_none()
        if not product:
            raise NotFoundError(f"Product {product_id} not found")
        return ProductDetail.model_validate(product)

    async def get_price_history(
        self, product_id: UUID, page: int, page_size: int
    ) -> PriceHistoryResponse:
        total = (await self.db.execute(
            select(func.count())
            .select_from(PriceHistory)
            .where(PriceHistory.product_id == product_id)
        )).scalar_one()

        stmt = (
            select(PriceHistory)
            .where(PriceHistory.product_id == product_id)
            .order_by(desc(PriceHistory.recorded_at))
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
        rows = (await self.db.execute(stmt)).scalars().all()

        return PriceHistoryResponse(
            product_id=product_id,
            items=[PriceHistoryItem.model_validate(r) for r in rows],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
        )

    async def get_change_history(
        self, product_id: UUID, page: int, page_size: int
    ) -> ChangeHistoryResponse:
        total = (await self.db.execute(
            select(func.count())
            .select_from(ProductChange)
            .where(ProductChange.product_id == product_id)
        )).scalar_one()

        stmt = (
            select(ProductChange)
            .where(ProductChange.product_id == product_id)
            .order_by(desc(ProductChange.created_at))
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
        rows = (await self.db.execute(stmt)).scalars().all()

        return ChangeHistoryResponse(
            product_id=product_id,
            items=[ChangeHistoryItem.model_validate(r) for r in rows],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
        )