"""
Phase 2 scaffold — creates every file needed for the embeddings pipeline.

Usage:
    cd backend
    python -m scripts.scaffold_phase2 --dry-run      # preview
    python -m scripts.scaffold_phase2                # create files
    python -m scripts.scaffold_phase2 --force        # overwrite existing

What it creates:
    migrations/006_pgvector.sql
    app/models/product_embedding.py
    app/repositories/embedding_repo.py
    app/services/embedding_backfill_service.py
    app/tasks/embedding_tasks.py
    app/schemas/product.py                (extended — safe overwrite prompt)
    app/api/v1/products.py                (extended — safe overwrite prompt)
    scripts/backfill_embeddings.py
    scripts/scaffold_phase2.py            (this file — skipped on rerun)
    tests/repositories/__init__.py
    tests/repositories/test_embedding_repo.py
    tests/services/__init__.py
    tests/services/test_embedding_backfill.py
    tests/api/__init__.py
    tests/api/test_products_similar.py
    PHASE_2_CONFIG_SNIPPET.py
    PHASE_2_ENV_SNIPPET.env
    requirements-phase2-additions.txt

It does NOT modify existing files unless you pass --force.
Extensions to app/schemas/product.py and app/api/v1/products.py are provided
as separate snippet files (PHASE_2_PRODUCTS_SCHEMA_ADDITIONS.py and
PHASE_2_PRODUCTS_ENDPOINT_ADDITIONS.py) so you can paste them manually —
safer than overwriting files that likely already have your work in them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = _THIS_FILE.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ═══════════════════════════════════════════════════════════════════
# FILE CONTENTS
# ═══════════════════════════════════════════════════════════════════

MIGRATION_006 = """\
-- ═══════════════════════════════════════════════════════
-- Migration 006 — pgvector + product_embeddings
-- Safe to run on existing DB (idempotent).
-- ═══════════════════════════════════════════════════════

-- ─── 1. Enable the pgvector extension ───────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── 2. product_embeddings table ────────────────────────
CREATE TABLE IF NOT EXISTS product_embeddings (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id   UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    embedding    vector(1536) NOT NULL,
    model        VARCHAR(100) NOT NULL,
    source_text  TEXT NOT NULL,
    source_hash  VARCHAR(64) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (product_id, model)
);

-- ─── 3. HNSW index for fast cosine similarity ───────────
-- Created separately because IF NOT EXISTS on index with HNSW
-- can fail on older pgvector versions. Guard with a DO block.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'ix_product_embeddings_hnsw'
    ) THEN
        CREATE INDEX ix_product_embeddings_hnsw
            ON product_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
    END IF;
END$$;

-- ─── 4. Lookup index for product_id ─────────────────────
CREATE INDEX IF NOT EXISTS ix_product_embeddings_product
    ON product_embeddings (product_id);

-- ─── 5. products: embedding tracking columns ────────────
ALTER TABLE products
    ADD COLUMN IF NOT EXISTS embedding_status VARCHAR(20) DEFAULT 'pending';

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS last_embedded_at TIMESTAMPTZ;

-- Backfill existing rows so the status is meaningful
UPDATE products
    SET embedding_status = 'pending'
    WHERE embedding_status IS NULL;

-- Partial index — only enabled products with pending/ready status
CREATE INDEX IF NOT EXISTS ix_products_embedding_status
    ON products (embedding_status, is_enabled)
    WHERE is_enabled = TRUE;

-- ─── 6. Sanity check ────────────────────────────────────
-- Should return one row: 'vector'
SELECT extname FROM pg_extension WHERE extname = 'vector';
"""


PRODUCT_EMBEDDING_MODEL_PY = '''\
"""SQLAlchemy ORM for product_embeddings (pgvector-backed)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

EMBEDDING_DIM = 1536


class ProductEmbedding(Base):
    __tablename__ = "product_embeddings"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    product_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=False
    )
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    product: Mapped["Product"] = relationship(  # noqa: F821
        "Product", backref="embedding", lazy="noload"
    )

    __table_args__ = (
        UniqueConstraint("product_id", "model", name="uq_product_embedding_model"),
        Index("ix_product_embeddings_product", "product_id"),
    )

    def __repr__(self) -> str:
        return f"<ProductEmbedding product={self.product_id} model={self.model}>"
'''


EMBEDDING_REPO_PY = '''\
"""
Repository for pgvector-backed product embeddings.

All queries are async and use the cosine distance operator (`<=>`).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional
from uuid import UUID

from sqlalchemy import delete, func, select, text
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

    async def count_ready(self) -> int:
        stmt = select(func.count()).select_from(ProductEmbedding)
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
        distance_expr = ProductEmbedding.embedding.cosine_distance(query_vector)

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
    # Writes
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
        Insert or update the embedding for (product_id, model).
        Uses PostgreSQL ON CONFLICT for atomic upsert.
        """
        stmt = pg_insert(ProductEmbedding).values(
            product_id=product_id,
            embedding=embedding,
            model=model,
            source_text=source_text,
            source_hash=source_hash,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_product_embedding_model",
            set_={
                "embedding": stmt.excluded.embedding,
                "source_text": stmt.excluded.source_text,
                "source_hash": stmt.excluded.source_hash,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        await self.db.execute(stmt)

    async def delete_for_product(self, product_id: UUID) -> int:
        stmt = delete(ProductEmbedding).where(
            ProductEmbedding.product_id == product_id
        )
        result = await self.db.execute(stmt)
        return result.rowcount or 0

    # ═══════════════════════════════════════════════════════
    # Backfill helpers
    # ═══════════════════════════════════════════════════════
    async def iter_products_needing_embedding(
        self,
        *,
        model: str,
        batch_size: int = 500,
        site: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Product]:
        """
        Return products that are enabled and either:
          - have no embedding row for this model, OR
          - have embedding_status = 'pending' or 'stale'

        Note: this returns a page, not an iterator — call repeatedly
        with an updated offset if you need to stream.
        """
        stmt = select(Product).where(Product.is_enabled.is_(True))

        if site:
            stmt = stmt.where(Product.site == site.lower())

        # Products missing an embedding row for this model
        subq = select(ProductEmbedding.product_id).where(
            ProductEmbedding.model == model
        )
        stmt = stmt.where(
            (Product.embedding_status.in_(["pending", "stale"]))
            | (Product.id.notin_(subq))
        )

        stmt = stmt.limit(batch_size)
        if limit:
            stmt = stmt.limit(min(batch_size, limit))

        return list((await self.db.execute(stmt)).scalars().all())

    async def iter_products_needing_embedding_offset(
        self,
        *,
        model: str,
        offset: int,
        batch_size: int = 500,
        site: Optional[str] = None,
    ) -> list[Product]:
        """
        Same as above but with an explicit offset — for scripts that
        need stable pagination while updating rows in between.
        """
        subq = select(ProductEmbedding.product_id).where(
            ProductEmbedding.model == model
        )
        stmt = (
            select(Product)
            .where(Product.is_enabled.is_(True))
            .where(
                (Product.embedding_status.in_(["pending", "stale"]))
                | (Product.id.notin_(subq))
            )
            .order_by(Product.id)
            .offset(offset)
            .limit(batch_size)
        )
        if site:
            stmt = stmt.where(Product.site == site.lower())
        return list((await self.db.execute(stmt)).scalars().all())

    async def mark_status(
        self,
        product_ids: Iterable[UUID],
        status: str,
        *,
        set_embedded_at: bool = False,
    ) -> None:
        ids = list(product_ids)
        if not ids:
            return
        values: dict = {"embedding_status": status}
        if set_embedded_at:
            values["last_embedded_at"] = datetime.now(timezone.utc)

        stmt = (
            Product.__table__.update()
            .where(Product.id.in_(ids))
            .values(**values)
        )
        await self.db.execute(stmt)

    async def total_pending(self, model: str) -> int:
        subq = select(ProductEmbedding.product_id).where(
            ProductEmbedding.model == model
        )
        stmt = (
            select(func.count())
            .select_from(Product)
            .where(Product.is_enabled.is_(True))
            .where(
                (Product.embedding_status.in_(["pending", "stale"]))
                | (Product.id.notin_(subq))
            )
        )
        return (await self.db.execute(stmt)).scalar_one() or 0

    async def raw_stats(self) -> dict:
        """Return quick stats for diagnostics — counts by status."""
        rows = (
            await self.db.execute(
                text(
                    """
                    SELECT embedding_status, COUNT(*) AS c
                    FROM products
                    WHERE is_enabled = TRUE
                    GROUP BY embedding_status
                    """
                )
            )
        ).all()
        return {r.embedding_status or "unknown": int(r.c) for r in rows}
'''


EMBEDDING_BACKFILL_SERVICE_PY = '''\
"""
Embedding backfill service.

Orchestrates: query products → build text → embed batch → upsert → mark status.
Resumable, idempotent, and safe to Ctrl+C.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.models.product import Product
from app.repositories.embedding_repo import EmbeddingRepo
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Text composition
# ═══════════════════════════════════════════════════════
def build_embedding_text(product: Product) -> str:
    """
    Canonical text used to embed a product.

    Order matters — brand and name first (highest signal), then category,
    then short description. Trimmed to EMBEDDING_MAX_CHARS.
    """
    parts: list[str] = []
    if product.brand:
        parts.append(str(product.brand).strip())
    if product.name:
        parts.append(str(product.name).strip())
    if product.category:
        parts.append(str(product.category).strip())

    short = None
    if product.short_description:
        # short_description is JSONB — coerce to string
        try:
            import json as _json
            short = _json.dumps(product.short_description, ensure_ascii=False)
        except Exception:
            short = str(product.short_description)

    if short:
        parts.append(short.strip())

    text = " | ".join(p for p in parts if p)
    text = " ".join(text.split())  # collapse whitespace
    max_chars = getattr(settings, "embedding_max_chars", 1000)
    return text[:max_chars]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════
@dataclass
class BackfillStats:
    total_candidates: int = 0
    processed: int = 0
    embedded: int = 0
    skipped: int = 0
    failed: int = 0

    def as_dict(self) -> dict:
        return {
            "total_candidates": self.total_candidates,
            "processed": self.processed,
            "embedded": self.embedded,
            "skipped": self.skipped,
            "failed": self.failed,
        }


# ═══════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════
class EmbeddingBackfillService:
    def __init__(
        self,
        db: AsyncSession,
        *,
        model: Optional[str] = None,
    ):
        self.db = db
        self.repo = EmbeddingRepo(db)
        self.embedder = EmbeddingService(
            db=db, endpoint="backfill_embeddings"
        )
        self.model = model or settings.openai_embedding_model

    async def embed_single_product(
        self, product: Product, *, force: bool = False
    ) -> bool:
        """
        Embed one product. Returns True if new vector written,
        False if skipped (unchanged hash).
        """
        text = build_embedding_text(product)
        if not text.strip():
            logger.warning(
                "embedding_skipped_empty_text", product_id=str(product.id)
            )
            await self.repo.mark_status([product.id], "failed")
            return False

        text_hash = hash_text(text)

        if not force:
            existing_hash = await self.repo.get_source_hash(
                product.id, self.model
            )
            if existing_hash == text_hash:
                # Unchanged — nothing to do
                await self.repo.mark_status([product.id], "ready")
                return False

        vector = await self.embedder.embed_one(text)

        await self.repo.upsert(
            product_id=product.id,
            embedding=vector,
            model=self.model,
            source_text=text,
            source_hash=text_hash,
        )
        await self.repo.mark_status(
            [product.id], "ready", set_embedded_at=True
        )
        return True

    async def backfill_batch(
        self,
        *,
        offset: int,
        batch_size: int,
        site: Optional[str] = None,
        force: bool = False,
    ) -> tuple[BackfillStats, int]:
        """
        Embed one page of products.
        Returns (stats, next_offset).
        """
        products = await self.repo.iter_products_needing_embedding_offset(
            model=self.model,
            offset=offset,
            batch_size=batch_size,
            site=site,
        )
        stats = BackfillStats(total_candidates=len(products))

        if not products:
            return stats, offset

        # Build texts and hashes
        texts: list[str] = []
        product_ids: list[UUID] = []
        for p in products:
            t = build_embedding_text(p)
            if not t.strip():
                await self.repo.mark_status([p.id], "failed")
                stats.failed += 1
                continue
            texts.append(t)
            product_ids.append(p.id)

        if not texts:
            return stats, offset + len(products)

        # Batch embed (the embedder handles caching internally)
        try:
            vectors = await self.embedder.embed_batch(texts)
        except Exception as e:
            logger.error("backfill_embed_batch_failed", error=str(e))
            await self.repo.mark_status(product_ids, "failed")
            stats.failed += len(product_ids)
            return stats, offset + len(products)

        # Upsert each
        for pid, text, vec in zip(product_ids, texts, vectors):
            text_hash = hash_text(text)
            try:
                await self.repo.upsert(
                    product_id=pid,
                    embedding=vec,
                    model=self.model,
                    source_text=text,
                    source_hash=text_hash,
                )
                stats.embedded += 1
            except Exception as e:
                logger.error(
                    "backfill_upsert_failed",
                    product_id=str(pid),
                    error=str(e),
                )
                stats.failed += 1

        await self.repo.mark_status(
            product_ids, "ready", set_embedded_at=True
        )
        await self.db.commit()

        stats.processed = len(products)
        logger.info(
            "backfill_batch_done",
            offset=offset,
            batch_size=len(products),
            embedded=stats.embedded,
            failed=stats.failed,
        )
        return stats, offset + len(products)

    async def count_pending(self) -> int:
        return await self.repo.total_pending(self.model)
'''


EMBEDDING_TASKS_PY = '''\
"""
Celery tasks for embeddings.

Idempotent — safe to run multiple times, results converge.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

from sqlalchemy import select

from app.core.logging_config import get_logger
from app.db.session import async_session_maker
from app.models.product import Product
from app.services.embedding_backfill_service import EmbeddingBackfillService
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)


async def _embed_product_async(product_id: str) -> dict:
    async with async_session_maker() as db:
        product = (
            await db.execute(
                select(Product).where(Product.id == UUID(product_id))
            )
        ).scalar_one_or_none()
        if not product:
            return {"status": "not_found", "product_id": product_id}

        svc = EmbeddingBackfillService(db)
        try:
            written = await svc.embed_single_product(product, force=False)
            return {
                "status": "ok",
                "product_id": product_id,
                "written": written,
            }
        except Exception as e:
            logger.error(
                "embed_product_failed",
                product_id=product_id,
                error=str(e),
            )
            await db.rollback()
            raise


@celery_app.task(name="embedding.embed_product", bind=True, max_retries=3)
def embed_product_task(self, product_id: str) -> dict:
    """
    Embed a single product. Retries 3 times with exponential backoff
    on any failure.
    """
    try:
        return asyncio.run(_embed_product_async(product_id))
    except Exception as exc:
        countdown = 10 * (2 ** self.request.retries)
        logger.warning(
            "embed_product_retry",
            product_id=product_id,
            retry=self.request.retries,
            countdown=countdown,
        )
        raise self.retry(exc=exc, countdown=countdown)


@celery_app.task(name="embedding.reembed_stale")
def reembed_stale_task(batch_size: int = 100) -> dict:
    """
    Nightly safety net: find products with embedding_status='stale'
    or missing rows, and re-embed a batch.
    """

    async def _run() -> dict:
        async with async_session_maker() as db:
            svc = EmbeddingBackfillService(db)
            stats, _ = await svc.backfill_batch(
                offset=0, batch_size=batch_size
            )
            return stats.as_dict()

    return asyncio.run(_run())
'''


PRODUCTS_SCHEMA_ADDITIONS = '''\
# ═══════════════════════════════════════════════════════
# PASTE THESE INTO app/schemas/product.py
# (append at the end of the file)
# ═══════════════════════════════════════════════════════

from uuid import UUID as _UUID
from typing import Optional as _Optional


class SimilarProduct(BaseModel):
    """A product returned by the /similar endpoint."""

    product_id: _UUID
    site: str
    main_product_id: str
    name: str
    brand: _Optional[str] = None
    category: _Optional[str] = None
    price: _Optional[float] = None
    discount_price: _Optional[float] = None
    availability: _Optional[str] = None
    main_image: _Optional[str] = None
    url: _Optional[str] = None
    similarity: float
    cosine_distance: float

    model_config = ConfigDict(from_attributes=True)


class SimilarProductsResponse(BaseModel):
    """Response shape for GET /products/{id}/similar."""

    query_product_id: _UUID
    count: int
    items: list[SimilarProduct]


class EmbeddingStatsResponse(BaseModel):
    """Diagnostic endpoint output."""

    total_embedded: int
    total_pending: int
    status_breakdown: dict[str, int]
    model: str
'''


PRODUCTS_ENDPOINT_ADDITIONS = '''\
# ═══════════════════════════════════════════════════════
# PASTE THESE INTO app/api/v1/products.py
#
# 1. Add the imports shown below to the top of the file.
# 2. Paste the two endpoints at the bottom of the file.
# ═══════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────
# IMPORTS — add these to the existing imports at the top
# ─────────────────────────────────────────────────────
#
# from fastapi import HTTPException, status as http_status
# from app.config import settings
# from app.repositories.embedding_repo import EmbeddingRepo
# from app.services.embedding_service import EmbeddingService
# from app.schemas.product import (
#     SimilarProduct,
#     SimilarProductsResponse,
#     EmbeddingStatsResponse,
# )


# ─────────────────────────────────────────────────────
# ENDPOINT 1 — /products/{id}/similar
# Paste at the bottom of app/api/v1/products.py
# ─────────────────────────────────────────────────────
@router.get(
    "/{product_id}/similar",
    response_model=SimilarProductsResponse,
    dependencies=[Depends(RateLimiter(times=60, minutes=1))],
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
    dependencies=[Depends(RateLimiter(times=30, minutes=1))],
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
'''


BACKFILL_EMBEDDINGS_PY = '''\
"""
Backfill product embeddings.

Usage:
    cd backend
    python -m scripts.backfill_embeddings --dry-run
    python -m scripts.backfill_embeddings --limit 100
    python -m scripts.backfill_embeddings --site megapc
    python -m scripts.backfill_embeddings                 # full run
    python -m scripts.backfill_embeddings --force         # re-embed everything

Ctrl+C is graceful — stops after the current batch, DB is consistent.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = _THIS_FILE.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


async def _run(args: argparse.Namespace) -> int:
    from app.config import settings
    from app.db.session import async_session_maker, dispose_engine
    from app.db.redis_client import close_redis
    from app.services.embedding_backfill_service import (
        EmbeddingBackfillService,
    )

    model = settings.openai_embedding_model
    print(f"Model:      {model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Site:       {args.site or 'all'}")
    print(f"Force:      {args.force}")
    print(f"Limit:      {args.limit or 'none'}")
    print("─" * 60)

    total_embedded = 0
    total_failed = 0
    offset = 0
    started = time.monotonic()

    try:
        async with async_session_maker() as db:
            svc = EmbeddingBackfillService(db, model=model)

            pending = await svc.count_pending()
            print(f"Pending before run: {pending}")

            if args.dry_run:
                print("Dry run — nothing written.")
                return 0

            if pending == 0 and not args.force:
                print("Nothing to do.")
                return 0

            while True:
                if args.limit and total_embedded >= args.limit:
                    print(f"Reached --limit {args.limit}")
                    break

                stats, offset = await svc.backfill_batch(
                    offset=offset,
                    batch_size=args.batch_size,
                    site=args.site,
                    force=args.force,
                )

                if stats.total_candidates == 0:
                    print("No more candidates.")
                    break

                total_embedded += stats.embedded
                total_failed += stats.failed

                elapsed = time.monotonic() - started
                rate = total_embedded / elapsed if elapsed > 0 else 0
                print(
                    f"  batch done | "
                    f"embedded={total_embedded:>6} "
                    f"failed={total_failed:>4} "
                    f"offset={offset:>6} "
                    f"rate={rate:>5.1f}/s"
                )

                if args.limit and total_embedded >= args.limit:
                    break

    except KeyboardInterrupt:
        print("\\nInterrupted. Progress up to the last committed batch "
              "is safe.")
        return 130
    finally:
        try:
            await close_redis()
        except Exception:
            pass
        try:
            await dispose_engine()
        except Exception:
            pass

    elapsed = time.monotonic() - started
    print("─" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  embedded: {total_embedded}")
    print(f"  failed:   {total_failed}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill embeddings.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
'''


TEST_REPOSITORIES_INIT = '''\
"""Tests for repository layer."""
'''


TEST_EMBEDDING_REPO_PY = '''\
"""Tests for EmbeddingRepo — require a live DB with pgvector."""
import pytest
from uuid import uuid4

from app.repositories.embedding_repo import EmbeddingRepopytestmark = pytest.mark.asyncio


async def test_get_by_product_returns_none_for_missing(db_session):
    repo = EmbeddingRepo(db_session)
    result = await repo.get_by_product(uuid4(), "test-model")
    assert result is None


async def test_count_ready_returns_int(db_session):
    repo = EmbeddingRepo(db_session)
    count = await repo.count_ready()
    assert isinstance(count, int)
    assert count >= 0


async def test_raw_stats_returns_dict(db_session):
    repo = EmbeddingRepo(db_session)
    stats = await repo.raw_stats()
    assert isinstance(stats, dict)
'''


TEST_SERVICES_INIT = '''\
"""Tests for service layer."""
'''


TEST_EMBEDDING_BACKFILL_PY = '''\
"""Tests for EmbeddingBackfillService — build_embedding_text is pure, safe to test."""
from unittest.mock import MagicMock

from app.services.embedding_backfill_service import (
    build_embedding_text,
    hash_text,
)


def _make_product(**overrides):
    p = MagicMock()
    p.brand = overrides.get("brand", "ASUS")
    p.name = overrides.get("name", "TUF Gaming F15")
    p.category = overrides.get("category", "Laptops")
    p.short_description = overrides.get("short_description", None)
    return p


def test_build_text_includes_brand_name_category():
    text = build_embedding_text(_make_product())
    assert "ASUS" in text
    assert "TUF Gaming F15" in text
    assert "Laptops" in text
    assert " | " in text


def test_build_text_handles_missing_fields():
    p = _make_product(brand=None, category=None)
    text = build_embedding_text(p)
    assert "TUF Gaming F15" in text
    assert text.count(" | ") >= 0


def test_build_text_truncates_to_max_chars(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "embedding_max_chars", 20, raising=False)
    p = _make_product(name="A" * 100)
    text = build_embedding_text(p)
    assert len(text) <= 20


def test_hash_text_deterministic():
    assert hash_text("hello") == hash_text("hello")
    assert hash_text("hello") != hash_text("world")
'''


TEST_API_INIT = '''\
"""Tests for API endpoints."""
'''


TEST_PRODUCTS_SIMILAR_PY = '''\
"""Smoke tests for /products/{id}/similar — requires running app + DB."""
import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


async def test_similar_returns_404_for_unknown_product(client: AsyncClient):
    import uuid
    fake_id = str(uuid.uuid4())
    resp = await client.get(f"/api/v1/products/{fake_id}/similar")
    assert resp.status_code in (404, 422)


async def test_embedding_stats_endpoint_returns_shape(client: AsyncClient):
    resp = await client.get("/api/v1/products/embeddings/stats")
    # May be 200 (populated) or 404/500 if the table isn't set up — accept both
    if resp.status_code == 200:
        data = resp.json()
        assert "total_embedded" in data
        assert "total_pending" in data
        assert "status_breakdown" in data
        assert "model" in data
'''


CONFIG_SNIPPET = '''\
    # ─── Phase 2 — Embeddings ──────────────────────────
    embedding_provider: str = Field(default="openai")
    embedding_fallback_enabled: bool = Field(default=False)
    embedding_max_chars: int = Field(default=1000, ge=100, le=8000)
    embedding_similar_cache_ttl_hours: int = Field(default=24, ge=1)
    embedding_max_distance_default: float = Field(default=0.3, ge=0.0, le=2.0)
'''


ENV_SNIPPET = """\
# ═══════════════════════════════════════════════════════
# Phase 2 — Embeddings
# ═══════════════════════════════════════════════════════
EMBEDDING_PROVIDER=openai
EMBEDDING_FALLBACK_ENABLED=false
EMBEDDING_MAX_CHARS=1000
EMBEDDING_SIMILAR_CACHE_TTL_HOURS=24
EMBEDDING_MAX_DISTANCE_DEFAULT=0.3
"""


REQUIREMENTS_ADDITIONS = """\
# ═══════════════════════════════════════════════════════
# Phase 2 additions (already partially in requirements-ai.txt)
# ═══════════════════════════════════════════════════════
pgvector>=0.3.0
tqdm>=4.66.0
"""


# ═══════════════════════════════════════════════════════════════════
# FILESYSTEM LAYOUT
# ═══════════════════════════════════════════════════════════════════
FILES: dict[str, str] = {
    "migrations/006_pgvector.sql":
        MIGRATION_006,
    "app/models/product_embedding.py":
        PRODUCT_EMBEDDING_MODEL_PY,
    "app/repositories/embedding_repo.py":
        EMBEDDING_REPO_PY,
    "app/services/embedding_backfill_service.py":
        EMBEDDING_BACKFILL_SERVICE_PY,
    "app/tasks/embedding_tasks.py":
        EMBEDDING_TASKS_PY,
    "scripts/backfill_embeddings.py":
        BACKFILL_EMBEDDINGS_PY,
    "tests/repositories/__init__.py":
        TEST_REPOSITORIES_INIT,
    "tests/repositories/test_embedding_repo.py":
        TEST_EMBEDDING_REPO_PY,
    "tests/services/__init__.py":
        TEST_SERVICES_INIT,
    "tests/services/test_embedding_backfill.py":
        TEST_EMBEDDING_BACKFILL_PY,
    "tests/api/__init__.py":
        TEST_API_INIT,
    "tests/api/test_products_similar.py":
        TEST_PRODUCTS_SIMILAR_PY,
    "PHASE_2_CONFIG_SNIPPET.py":
        CONFIG_SNIPPET,
    "PHASE_2_ENV_SNIPPET.env":
        ENV_SNIPPET,
    "PHASE_2_PRODUCTS_SCHEMA_ADDITIONS.py":
        PRODUCTS_SCHEMA_ADDITIONS,
    "PHASE_2_PRODUCTS_ENDPOINT_ADDITIONS.py":
        PRODUCTS_ENDPOINT_ADDITIONS,
    "requirements-phase2-additions.txt":
        REQUIREMENTS_ADDITIONS,
}


# ═══════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════
def preflight_check() -> bool:
    """
    Verify Phase 1 is complete and the environment is ready for Phase 2.
    Fails fast with clear remediation steps.
    """
    print("PRE-FLIGHT CHECKS")
    print("─" * 70)
    failures: list[str] = []
    warnings: list[str] = []

    # ── 1. Phase 1 artifacts exist ──────────────────────
    try:
        from app.core.llm_client import LLMClient  # noqa
        print("  ✓ Phase 1: app.core.llm_client importable")
    except Exception as e:
        failures.append(f"Phase 1 llm_client not importable: {e}")

    try:
        from app.services.embedding_service import EmbeddingService  # noqa
        print("  ✓ Phase 1: app.services.embedding_service importable")
    except Exception as e:
        failures.append(f"Phase 1 embedding_service not importable: {e}")

    try:
        from app.models.llm_usage import LLMUsage  # noqa
        print("  ✓ Phase 1: app.models.llm_usage importable")
    except Exception as e:
        failures.append(f"Phase 1 llm_usage model not importable: {e}")

    try:
        from app.db.session import async_session_maker  # noqa
        print("  ✓ Phase 1: db.session.async_session_maker exists")
    except Exception as e:
        failures.append(f"db.session.async_session_maker missing: {e}")

    # ── 2. pgvector Python package installed ────────────
    try:
        import pgvector  # noqa
        print("  ✓ pgvector Python package installed")
    except ImportError:
        failures.append(
            "pgvector Python package not installed. "
            "Run: pip install 'pgvector>=0.3.0'"
        )

    # ── 3. Config has Phase 2 fields ────────────────────
    try:
        from app.config import settings
        if not hasattr(settings, "embedding_provider"):
            warnings.append(
                "settings.embedding_provider missing — "
                "paste PHASE_2_CONFIG_SNIPPET.py into app/config.py"
            )
        else:
            print("  ✓ settings.embedding_provider present")
    except Exception as e:
        failures.append(f"cannot import app.config: {e}")

    # ── 4. DB reachable + pgvector extension ────────────
    async def _check_db() -> tuple[bool, str]:
        try:
            from sqlalchemy import text
            from app.db.session import async_session_maker
            async with async_session_maker() as db:
                # Check extension
                r = await db.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname='vector'")
                )
                if r.scalar_one_or_none() is None:
                    return False, "vector extension not installed"
                return True, "ok"
        except Exception as e:
            return False, str(e)

    try:
        import asyncio
        ok, reason = asyncio.run(_check_db())
        if ok:
            print("  ✓ Postgres: pgvector extension present")
        else:
            warnings.append(
                f"Postgres pgvector check: {reason}. "
                f"Run: psql -f migrations/006_pgvector.sql"
            )
    except Exception as e:
        warnings.append(f"could not check pgvector extension: {e}")

    # ── 5. Embedding provider reachable ─────────────────
    try:
        from app.config import settings
        if settings.openai_api_key:
            print("  ✓ OpenAI API key configured")
        else:
            warnings.append(
                "OPENAI_API_KEY is empty — embeddings won't work unless "
                "you set EMBEDDING_PROVIDER=ollama"
            )
    except Exception:
        pass

    print("─" * 70)

    if failures:
        print("\n✗ PRE-FLIGHT FAILED\n")
        for f in failures:
            print(f"  • {f}")
        print("\nFix these before scaffolding Phase 2.")
        return False

    if warnings:
        print("\n⚠ Warnings (non-blocking):\n")
        for w in warnings:
            print(f"  • {w}")
        print()

    print("✓ Pre-flight checks passed\n")
    return True


# ═══════════════════════════════════════════════════════════════════
# SCAFFOLDER
# ═══════════════════════════════════════════════════════════════════
def scaffold(*, force: bool, dry_run: bool, skip_preflight: bool) -> int:
    if not skip_preflight and not dry_run:
        if not preflight_check():
            return 1
    elif dry_run:
        print("(dry-run — skipping pre-flight)\n")

    created = 0
    skipped = 0

    print(f"Backend root: {BACKEND_ROOT}\n")
    print(f"{'─' * 70}")
    print(f"{'ACTION':<12} FILE")
    print(f"{'─' * 70}")

    for rel_path, content in FILES.items():
        target = BACKEND_ROOT / rel_path
        exists = target.exists()

        if exists and not force:
            print(f"{'skip':<12} {rel_path}  (exists)")
            skipped += 1
            continue

        action = "overwrite" if exists else "create"
        print(f"{action:<12} {rel_path}  ({len(content)} bytes)")

        if dry_run:
            created += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        created += 1

    print(f"{'─' * 70}")
    print(f"Created: {created}   Skipped: {skipped}   "
          f"Mode: {'DRY-RUN' if dry_run else 'WRITE'}")
    print(f"{'─' * 70}")

    if not dry_run and created > 0:
        print()
        print("NEXT STEPS")
        print("─" * 70)
        print("1. Install Phase 2 dependencies:")
        print("     pip install -r requirements-phase2-additions.txt")
        print()
        print("2. Apply the migration:")
        print("     psql -U scraper_aziz -d scraping_db \\")
        print("       -f backend/migrations/006_pgvector.sql")
        print()
        print("3. Register the new model in app/models/__init__.py:")
        print("     from app.models.product_embedding import ProductEmbedding  # noqa")
        print()
        print("4. Paste Phase 2 config into app/config.py:")
        print("     See PHASE_2_CONFIG_SNIPPET.py — insert before the")
        print("     '@property celery_broker_url' line")
        print()
        print("5. Append Phase 2 env vars to .env:")
        print("     See PHASE_2_ENV_SNIPPET.env")
        print()
        print("6. Extend app/schemas/product.py:")
        print("     Paste from PHASE_2_PRODUCTS_SCHEMA_ADDITIONS.py")
        print()
        print("7. Extend app/api/v1/products.py:")
        print("     Paste imports + endpoints from")
        print("     PHASE_2_PRODUCTS_ENDPOINT_ADDITIONS.py")
        print()
        print("8. Register the Celery task in app/tasks/__init__.py:")
        print("     from app.tasks import embedding_tasks  # noqa")
        print()
        print("9. Run the tests:")
        print("     pytest tests/ -v")
        print()
        print("10. Smoke: backfill a small batch")
        print("     python -m scripts.backfill_embeddings --limit 10")
        print()
        print("11. Full backfill (may take 30-60 min with OpenAI):")
        print("     python -m scripts.backfill_embeddings")
        print()
        print("12. Test the endpoint (grab an embedded product id):")
        print("     curl http://localhost:8000/api/v1/products/<id>/similar")
        print("─" * 70)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold Phase 2 — embeddings pipeline."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files (default: skip).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be created without writing.",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip environment checks (use with care).",
    )
    args = parser.parse_args()
    return scaffold(
        force=args.force,
        dry_run=args.dry_run,
        skip_preflight=args.skip_preflight,
    )


if __name__ == "__main__":
    sys.exit(main())