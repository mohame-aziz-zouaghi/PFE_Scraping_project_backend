"""Product endpoints."""
from http.client import HTTPException
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import rate_limit_anon, rate_limit_user
from app.db.session import get_db
from app.repositories.embedding_repo import EmbeddingRepo
from app.schemas.product import (
    ChangeHistoryResponse,
    PaginatedProducts,
    PriceHistoryResponse,
    ProductDetail,
    SimilarProductsResponse,
)
from app.services.product_service import ProductService
from fastapi import HTTPException, status as http_status
from app.config import settings
from app.repositories.embedding_repo import EmbeddingRepo
from app.services.embedding_service import EmbeddingService
from app.schemas.product import (
    SimilarProduct,
    SimilarProductsResponse,
    EmbeddingStatsResponse,
)

router = APIRouter()


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
    availability: Optional[str] = Query(None, max_length=20),
    brand: Optional[str] = Query(None, max_length=200),
    category: Optional[str] = Query(None, max_length=200),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    sort_by: str = Query(
        "last_updated_at",
        pattern="^(name|price|brand|category|last_updated_at|last_checked)$",
    ),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    db: AsyncSession = Depends(get_db),
):
    """List products with filters + pagination."""
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
    )


@router.get("/{product_id}", response_model=ProductDetail,
                dependencies=[rate_limit_anon(times=200, seconds=60)])
async def get_product(
    request: Request,
    product_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get single product."""
    service = ProductService(db)
    return await service.get_product(product_id)


@router.get("/{product_id}/history", response_model=PriceHistoryResponse,
                dependencies=[rate_limit_anon(times=100, seconds=60)])
async def get_price_history(
    request: Request,
    product_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Get price history for a product."""
    service = ProductService(db)
    return await service.get_price_history(product_id, page, page_size)


@router.get("/{product_id}/changes", response_model=ChangeHistoryResponse,
                dependencies=[rate_limit_anon(times=100, seconds=60)])
async def get_change_history(
    request: Request,
    product_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Get change history for a product."""
    service = ProductService(db)
    return await service.get_change_history(product_id, page, page_size)




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
    """
    Return products semantically similar to the given one, ranked by
    cosine distance. Uses pgvector HNSW index.
    """
    repo = EmbeddingRepo(db)

    # 1. Fetch the source product's embedding
    source_embedding = await repo.get_by_product(
        product_id, settings.openai_embedding_model
    )
    if source_embedding is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="Embedding not found for this product yet",
        )

    # 2. Nearest-neighbour lookup
    neighbours = await repo.find_similar(
        query_vector=source_embedding.embedding,
        model=source_embedding.model,
        exclude_product_id=product_id,
        limit=limit,
        max_distance=max_distance,
        only_enabled=True,
    )

    # 3. Shape the response
    items = [
        SimilarProduct(
            product_id=n["product"].id,
            site=n["product"].site,
            main_product_id=n["product"].main_product_id,
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


# ─────────────────────────────────────────────────────
# ENDPOINT 2 — /products/embeddings/stats
# Paste at the bottom of app/api/v1/products.py
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
    """Diagnostic — how many products are embedded, how many pending."""
    repo = EmbeddingRepo(db)
    model = settings.openai_embedding_model

    total_embedded = await repo.count_ready()
    total_pending = await repo.total_pending(model)
    breakdown = await repo.raw_stats()

    return EmbeddingStatsResponse(
        total_embedded=total_embedded,
        total_pending=total_pending,
        status_breakdown=breakdown,
        model=model,
    )
