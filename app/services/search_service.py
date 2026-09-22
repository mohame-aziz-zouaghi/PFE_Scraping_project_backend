"""
Search service — text search with optional semantic fallback.

Text search uses PostgreSQL ILIKE. Fast and reliable.
Semantic search requires pgvector embeddings on `products.embedding`.
"""
import time
from typing import Optional

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product import Product
from app.schemas.search import (
    SearchResponse,
    SearchResult,
    SemanticSearchRequest,
    SuggestionsResponse,
    SuggestionItem,
    TextSearchRequest,
)


class SearchService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── Text search ────────────────────────────────────
    async def text_search(self, payload: TextSearchRequest) -> SearchResponse:
        start = time.perf_counter()

        q = payload.query.strip()
        like_pattern = f"%{q}%"

        # Base filters
        conditions = [Product.is_enabled == True]
        if payload.site:
            conditions.append(Product.site == payload.site.lower())
        if payload.brand:
            conditions.append(Product.brand == payload.brand)
        if payload.category:
            conditions.append(Product.category == payload.category)
        if payload.availability:
            conditions.append(Product.availability == payload.availability)
        if payload.min_price is not None:
            conditions.append(Product.price >= payload.min_price)
        if payload.max_price is not None:
            conditions.append(Product.price <= payload.max_price)

        # Text match
        conditions.append(
            or_(
                Product.name.ilike(like_pattern),
                Product.brand.ilike(like_pattern),
                Product.category.ilike(like_pattern),
            )
        )

        # Count
        count_stmt = (
            select(func.count()).select_from(Product).where(and_(*conditions))
        )
        total = (await self.db.execute(count_stmt)).scalar_one()

        # Ordering
        if payload.sort_by == "price":
            order = Product.price.asc()
        elif payload.sort_by == "price_desc":
            order = Product.price.desc()
        elif payload.sort_by == "recent":
            order = desc(Product.last_updated_at)
        else:
            # relevance — shorter names first (heuristic)
            order = func.length(Product.name).asc()

        stmt = (
            select(Product)
            .where(and_(*conditions))
            .order_by(order)
            .limit(payload.page_size)
            .offset((payload.page - 1) * payload.page_size)
        )
        rows = (await self.db.execute(stmt)).scalars().all()

        items = [self._to_result(r, score=1.0) for r in rows]
        took_ms = (time.perf_counter() - start) * 1000

        return SearchResponse(
            query=q,
            items=items,
            total=total,
            page=payload.page,
            page_size=payload.page_size,
            total_pages=(total + payload.page_size - 1) // payload.page_size,
            took_ms=round(took_ms, 2),
        )

    # ─── Suggestions ────────────────────────────────────
    async def suggestions(self, query: str, limit: int) -> SuggestionsResponse:
        q = query.strip()
        like_pattern = f"%{q}%"

        name_rows = (await self.db.execute(
            select(Product.name, func.count().label("cnt"))
            .where(Product.is_enabled == True)
            .where(Product.name.ilike(like_pattern))
            .group_by(Product.name)
            .order_by(desc("cnt"))
            .limit(limit)
        )).all()

        brand_rows = (await self.db.execute(
            select(Product.brand, func.count().label("cnt"))
            .where(Product.is_enabled == True)
            .where(Product.brand.ilike(like_pattern))
            .where(Product.brand.isnot(None))
            .group_by(Product.brand)
            .order_by(desc("cnt"))
            .limit(limit)
        )).all()

        cat_rows = (await self.db.execute(
            select(Product.category, func.count().label("cnt"))
            .where(Product.is_enabled == True)
            .where(Product.category.ilike(like_pattern))
            .where(Product.category.isnot(None))
            .group_by(Product.category)
            .order_by(desc("cnt"))
            .limit(limit)
        )).all()

        items: list[SuggestionItem] = []
        for name, cnt in name_rows:
            items.append(SuggestionItem(text=name, type="product", count=int(cnt)))
        for brand, cnt in brand_rows:
            items.append(SuggestionItem(text=brand, type="brand", count=int(cnt)))
        for cat, cnt in cat_rows:
            items.append(SuggestionItem(text=cat, type="category", count=int(cnt)))

        return SuggestionsResponse(query=q, items=items[:limit])

    # ─── Semantic search ────────────────────────────────
    async def semantic_search(
        self, payload: SemanticSearchRequest
    ) -> SearchResponse:
        """
        Semantic search using pgvector. Falls back to text search if
        embeddings are not available.
        """
        start = time.perf_counter()
        q = payload.query.strip()

        # Try to obtain an embedding for the query
        embedding_vector: Optional[list[float]] = None
        try:
            from app.core.embeddings import embed_text
            embedding_vector = await embed_text(q)
        except Exception:
            embedding_vector = None

        # Fallback: text search
        if embedding_vector is None:
            fallback = await self.text_search(
                TextSearchRequest(
                    query=q,
                    site=payload.site,
                    page=1,
                    page_size=payload.limit,
                )
            )
            fallback.took_ms = round((time.perf_counter() - start) * 1000, 2)
            return fallback

        # Vector search (cosine distance <=>)
        conditions = [Product.is_enabled == True, Product.embedding.isnot(None)]
        if payload.site:
            conditions.append(Product.site == payload.site.lower())

        similarity = 1 - Product.embedding.cosine_distance(embedding_vector)
        stmt = (
            select(Product, similarity.label("similarity"))
            .where(and_(*conditions))
            .where(similarity >= payload.min_similarity)
            .order_by(Product.embedding.cosine_distance(embedding_vector))
            .limit(payload.limit)
        )

        rows = (await self.db.execute(stmt)).all()

        items = [self._to_result(r[0], score=float(r[1] or 0)) for r in rows]
        took_ms = (time.perf_counter() - start) * 1000

        return SearchResponse(
            query=q,
            items=items,
            total=len(items),
            page=1,
            page_size=payload.limit,
            total_pages=1,
            took_ms=round(took_ms, 2),
        )

    # ─── Helper ─────────────────────────────────────────
    @staticmethod
    def _to_result(p: Product, score: float = 1.0) -> SearchResult:
        return SearchResult(
            product_id=p.id,
            site=p.site,
            name=p.name,
            brand=p.brand,
            category=p.category,
            price=float(p.price or 0),
            availability=p.availability,
            main_image=p.main_image,
            url=p.url,
            score=score,
        )