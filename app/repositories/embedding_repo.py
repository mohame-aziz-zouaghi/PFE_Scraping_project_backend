"""
Repository for pgvector-backed product embeddings.

product_embeddings is the source of truth for "does this product
have a vector?". products.embedding_status and
products.last_embedded_at are denormalized mirrors that exist so
queries like "show me all un-embedded products" don't need a
LEFT JOIN against product_embeddings.

The repo keeps the two tables in sync:

    upsert()             →  writes embedding + sets 'complete'
    delete_for_product() →  deletes embedding + sets 'pending'
    invalidate()         →  same as delete_for_product, scoped
                            optionally by model

All queries are async and use the cosine distance operator (`<=>`).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.models.product import Product
from app.models.product_embedding import ProductEmbedding

logger = get_logger(__name__)


class EmbeddingRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ═══════════════════════════════════════════════════════
    # Reads
    # ═══════════════════════════════════════════════════════
    async def get_by_product(
        self, product_id: UUID, model: str
    ) -> Optional[ProductEmbedding]:
        stmt = select(ProductEmbedding).where(
            ProductEmbedding.product_id == product_id,
            ProductEmbedding.model == model,
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def get_source_hash(
        self, product_id: UUID, model: str
    ) -> Optional[str]:
        stmt = select(ProductEmbedding.source_hash).where(
            ProductEmbedding.product_id == product_id,
            ProductEmbedding.model == model,
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def count_ready(self, model: Optional[str] = None) -> int:
        stmt = select(func.count()).select_from(ProductEmbedding)
        if model:
            stmt = stmt.where(ProductEmbedding.model == model)
        return (await self.db.execute(stmt)).scalar_one() or 0

    async def find_similar(
        self,
        *,
        query_vector: list[float],
        model: str,
        exclude_product_id: Optional[UUID] = None,
        limit: int = 10,
        max_distance: float = 0.3,
        only_enabled: bool = True,
    ) -> list[dict]:
        """
        Nearest-neighbour lookup by cosine distance.
        Returns a list of dicts with product fields + similarity + distance.
        """
        distance_expr = ProductEmbedding.embedding.cosine_distance(
            query_vector
        )

        stmt = (
            select(
                Product,
                distance_expr.label("distance"),
            )
            .join(ProductEmbedding, ProductEmbedding.product_id == Product.id)
            .where(ProductEmbedding.model == model)
            .where(distance_expr < max_distance)
        )

        if exclude_product_id is not None:
            stmt = stmt.where(Product.id != exclude_product_id)
        if only_enabled:
            stmt = stmt.where(Product.is_enabled.is_(True))

        stmt = stmt.order_by(distance_expr).limit(limit)

        rows = (await self.db.execute(stmt)).all()
        results: list[dict] = []
        for product, distance in rows:
            d = float(distance)
            results.append(
                {
                    "product": product,
                    "cosine_distance": d,
                    "similarity": round(1.0 - d, 6),
                }
            )
        return results

    # ═══════════════════════════════════════════════════════
    # Writes — single-product
    # ═══════════════════════════════════════════════════════
    async def upsert(
        self,
        *,
        product_id: UUID,
        embedding: list[float],
        model: str,
        source_text: str,
        source_hash: str,
    ) -> None:
        """
        Insert or update the embedding for (product_id, model)
        AND mirror the state into `products.embedding_status`
        and `products.last_embedded_at` — same transaction, so
        the two tables never disagree.

        Uses ON CONFLICT on the natural key (product_id, model) — the
        unique constraint created by migration 006.
        """
        now = datetime.now(timezone.utc)

        # ── 1. Write the embedding
        stmt = pg_insert(ProductEmbedding).values(
            product_id=product_id,
            embedding=embedding,
            model=model,
            source_text=source_text,
            source_hash=source_hash,
            created_at=now,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["product_id", "model"],
            set_={
                "embedding": stmt.excluded.embedding,
                "source_text": stmt.excluded.source_text,
                "source_hash": stmt.excluded.source_hash,
                "updated_at": now,
            },
        )
        await self.db.execute(stmt)

        # ── 2. Mirror into products (same transaction)
        await self.db.execute(
            update(Product)
            .where(Product.id == product_id)
            .values(
                embedding_status="complete",
                last_embedded_at=now,
            )
        )

    # ═══════════════════════════════════════════════════════
    # Writes — bulk
    # ═══════════════════════════════════════════════════════
    async def upsert_bulk(
        self,
        *,
        rows: list[dict],
        model: str,
    ) -> int:
        """
        Bulk insert-or-update embeddings and mirror the state into
        `products` in one transaction.

        `rows` must be a list of dicts, each with:
            product_id : UUID
            embedding  : list[float]
            source_text: str
            source_hash: str

        Returns the number of rows written.
        """
        if not rows:
            return 0

        now = datetime.now(timezone.utc)

        # ── 1. Bulk upsert embeddings
        values = [
            {
                "product_id": r["product_id"],
                "model": model,
                "embedding": r["embedding"],
                "source_text": r["source_text"],
                "source_hash": r["source_hash"],
                "created_at": now,
                "updated_at": now,
            }
            for r in rows
        ]

        stmt = pg_insert(ProductEmbedding).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["product_id", "model"],
            set_={
                "embedding": stmt.excluded.embedding,
                "source_text": stmt.excluded.source_text,
                "source_hash": stmt.excluded.source_hash,
                "updated_at": now,
            },
        )
        await self.db.execute(stmt)

        # ── 2. Bulk update products
        ids = [r["product_id"] for r in rows]
        await self.db.execute(
            update(Product)
            .where(Product.id.in_(ids))
            .values(
                embedding_status="complete",
                last_embedded_at=now,
            )
        )

        return len(rows)

    # ═══════════════════════════════════════════════════════
    # Deletes — flip status back to 'pending'
    # ═══════════════════════════════════════════════════════
    async def delete_for_product(self, product_id: UUID) -> int:
        """
        Delete all embeddings for a product and mark it 'pending'
        in products. Same transaction — either both happen or
        neither does.
        """
        stmt = delete(ProductEmbedding).where(
            ProductEmbedding.product_id == product_id
        )
        result = await self.db.execute(stmt)
        deleted = result.rowcount or 0

        if deleted > 0:
            await self.db.execute(
                update(Product)
                .where(Product.id == product_id)
                .values(
                    embedding_status="pending",
                    last_embedded_at=None,
                )
            )

        return deleted

    async def invalidate(
        self, product_id: UUID, model: Optional[str] = None
    ) -> int:
        """
        Mark a product's embeddings as stale by deleting them.
        The next backfill will re-embed and flip the status back
        to 'complete'.

        If `model` is None, deletes all models for the product.
        Only flips `products.embedding_status` when the last
        model for this product was deleted — otherwise there
        would still be an embedding present for a different
        model.
        """
        stmt = delete(ProductEmbedding).where(
            ProductEmbedding.product_id == product_id
        )
        if model:
            stmt = stmt.where(ProductEmbedding.model == model)
        result = await self.db.execute(stmt)
        deleted = result.rowcount or 0

        if deleted == 0:
            return 0

        # Did we wipe ALL models for this product, or just one?
        remaining = (await self.db.execute(
            select(func.count())
            .select_from(ProductEmbedding)
            .where(ProductEmbedding.product_id == product_id)
        )).scalar_one() or 0

        if remaining == 0:
            await self.db.execute(
                update(Product)
                .where(Product.id == product_id)
                .values(
                    embedding_status="pending",
                    last_embedded_at=None,
                )
            )

        return deleted

    # ═══════════════════════════════════════════════════════
    # Backfill helpers
    # ═══════════════════════════════════════════════════════
    async def iter_products_needing_embedding(
    self,
    *,
    model: str,
    batch_size: int = 500,
    site: Optional[str] = None,
) -> list[Product]:
        """
        Return the FIRST `batch_size` enabled products missing an embedding.
        Caller processes them, commits, and calls again — no offset.
        """
        subq = (
            select(ProductEmbedding.product_id)
            .where(ProductEmbedding.model == model)
            .where(ProductEmbedding.product_id == Product.id)
            .exists()
        )
        stmt = (
            select(Product)
            .where(Product.is_enabled.is_(True))
            .where(~subq)
            .order_by(Product.id)
            .limit(batch_size)
        )
        if site:
            stmt = stmt.where(Product.site == site.lower())
        return list((await self.db.execute(stmt)).scalars().all())

    async def total_missing(self, model: str) -> int:
        """Count enabled products with no embedding row for this model."""
        subq = (
            select(ProductEmbedding.product_id)
            .where(ProductEmbedding.model == model)
            .where(ProductEmbedding.product_id == Product.id)
            .exists()
        )
        stmt = (
            select(func.count())
            .select_from(Product)
            .where(Product.is_enabled.is_(True))
            .where(~subq)
        )
        return (await self.db.execute(stmt)).scalar_one() or 0

    async def total_embedded(self, model: Optional[str] = None) -> int:
        """Count embeddings, optionally for a specific model."""
        return await self.count_ready(model)

    async def fetch_missing_batch(
    self,
    *,
    model: str,
    limit: int,
    after_id: Optional[UUID] = None,
    site: Optional[str] = None,
) -> list[Product]:
        """
        Return up to `limit` enabled products that:
        - have no embedding for `model`
        - have id > after_id (cursor for stable pagination)

        Ordered by id so the caller can advance `after_id` monotonically.
        """
        subq = (
            select(ProductEmbedding.product_id)
            .where(ProductEmbedding.model == model)
            .where(ProductEmbedding.product_id == Product.id)
            .exists()
        )
        stmt = (
            select(Product)
            .where(Product.is_enabled.is_(True))
            .where(~subq)
            .order_by(Product.id.asc())
            .limit(limit)
        )
        if after_id is not None:
            stmt = stmt.where(Product.id > after_id)
        if site:
            stmt = stmt.where(Product.site == site.lower())

        return list((await self.db.execute(stmt)).scalars().all())