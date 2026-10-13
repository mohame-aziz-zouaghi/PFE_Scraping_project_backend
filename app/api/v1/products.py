"""Product endpoints."""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user_optional
from app.config import settings
from app.core.rate_limit import rate_limit_anon, rate_limit_user
from app.db.session import get_db
from app.repositories.embedding_repo import EmbeddingRepo
from app.schemas.product import (
    ChangeHistoryResponse,
    EmbeddingStatsResponse,
    PaginatedProducts,
    PriceHistoryResponse,
    ProductDetail,
    SimilarProduct,
    SimilarProductsResponse,
)
from app.services.product_service import ProductService

router = APIRouter()


# ═══════════════════════════════════════════════════════
# List
# ═══════════════════════════════════════════════════════
@router.get(
    "",
    response_model=PaginatedProducts,
    dependencies=[rate_limit_anon(times=100, seconds=60)],
)
async def list_products(
    request: Request,
    page: int = Query(1, ge=1, le=1000),
    page_size: int = Query(20, ge=1, le=100),
    site: Optional[str] = Query(None, max_length=50),
    availability: Optional[str] = Query(None, max_length=50),
    brand: Optional[str] = Query(None, max_length=200),
    category: Optional[str] = Query(None, max_length=200),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    sort_by: str = Query(
        "last_updated_at",
        pattern="^(name|price|brand|category|last_updated_at|last_checked)$",
    ),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    user=Depends(get_current_user_optional),   # <-- NEW
    db: AsyncSession = Depends(get_db),
):
    """List products.

    Anonymous visitors see at most
    `settings.public_products_per_category` items per view.
    Signed-in users see everything.
    """
    anonymous_cap: Optional[int] = (
        None
        if user is not None
        else settings.public_products_per_category
    )
    service = ProductService(db)
    return await service.list_products(
        page=page,
        page_size=page_size,
        site=site,
        availability=availability,
        brand=brand,
        category=category,
        min_price=min_price,
        max_price=max_price,
        sort_by=sort_by,
        sort_order=sort_order,
        anonymous_cap=anonymous_cap,
    )


# ═══════════════════════════════════════════════════════
# Detail — every column
# ═══════════════════════════════════════════════════════
@router.get(
    "/{product_id}",
    response_model=ProductDetail,
    dependencies=[rate_limit_anon(times=200, seconds=60)],
)
async def get_product(
    request: Request,
    product_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a single product with every column."""
    service = ProductService(db)
    return await service.get_product(product_id)


# ═══════════════════════════════════════════════════════
# History
# ═══════════════════════════════════════════════════════
@router.get(
    "/{product_id}/history",
    response_model=PriceHistoryResponse,
    dependencies=[rate_limit_anon(times=100, seconds=60)],
)
async def get_price_history(
    request: Request,
    product_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    service = ProductService(db)
    return await service.get_price_history(product_id, page, page_size)


@router.get(
    "/{product_id}/changes",
    response_model=ChangeHistoryResponse,
    dependencies=[rate_limit_anon(times=100, seconds=60)],
)
async def get_change_history(
    request: Request,
    product_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    service = ProductService(db)
    return await service.get_change_history(product_id, page, page_size)


# ═══════════════════════════════════════════════════════
# Similar products (embeddings)
# ═══════════════════════════════════════════════════════
@router.get(
    "/{product_id}/similar",
    response_model=SimilarProductsResponse,
    dependencies=[rate_limit_user(times=60, seconds=60)],
)
async def similar_products(
    request: Request,
    product_id: UUID,
    limit: int = Query(10, ge=1, le=50),
    max_distance: float = Query(0.3, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
):
    """Nearest neighbours by embedding cosine distance."""
    repo = EmbeddingRepo(db)
    model = settings.active_embedding_model

    source_embedding = await repo.get_by_product(product_id, model)
    if source_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Embedding not found for this product yet",
        )

    neighbours = await repo.find_similar(
        query_vector=source_embedding.embedding,
        model=source_embedding.model,
        exclude_product_id=product_id,
        limit=limit,
        max_distance=max_distance,
        only_enabled=True,
    )

    items = [
        SimilarProduct(
            product_id=n["product"].id,
            site=n["product"].site,
            main_product_id=n["product"].main_product_id,
            special_id=n["product"].special_id,
            name=n["product"].name,
            brand=n["product"].brand,
            category=n["product"].category,
            price=(
                float(n["product"].price)
                if n["product"].price is not None
                else None
            ),
            discount_price=(
                float(n["product"].discount_price)
                if n["product"].discount_price is not None
                else None
            ),
            availability=n["product"].availability,
            main_image=n["product"].main_image,
            url=n["product"].url,
            similarity=n["similarity"],
            cosine_distance=n["cosine_distance"],
        )
        for n in neighbours
    ]

    return SimilarProductsResponse(
        query_product_id=product_id,
        count=len(items),
        items=items,
    )


# ═══════════════════════════════════════════════════════
# Embedding stats (admin diagnostic)
# ═══════════════════════════════════════════════════════
@router.get(
    "/embeddings/stats",
    response_model=EmbeddingStatsResponse,
    dependencies=[rate_limit_user(times=30, seconds=60)],
)
async def embedding_stats(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    repo = EmbeddingRepo(db)
    model = settings.active_embedding_model

    total_embedded = await repo.count_ready()
    total_pending = await repo.total_pending(model)
    breakdown = await repo.raw_stats()

    return EmbeddingStatsResponse(
        total_embedded=total_embedded,
        total_pending=total_pending,
        status_breakdown=breakdown,
        model=model,
    )