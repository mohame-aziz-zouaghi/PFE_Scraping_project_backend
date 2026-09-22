"""
Embedding backfill service.

Orchestrates: query products → build text → embed batch → upsert into
product_embeddings.

Resumable, idempotent, safe to Ctrl+C.

The product_embeddings table is the single source of truth — there are
no mirror columns on products. "Needs embedding" == "no row exists."
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.models.product import Product
from app.repositories.embedding_repo import EmbeddingRepo
from app.services.embedding_service import (
    BackfillStats,
    EmbeddingService,
    build_embedding_text,
    hash_text,
)

logger = get_logger(__name__)


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
        # Pick the model based on the configured provider
        if model is not None:
            self.model = model
        elif settings.embedding_provider == "ollama":
            self.model = settings.ollama_embedding_model
        else:
            self.model = settings.openai_embedding_model

    # ═══════════════════════════════════════════════════════
    # Single-product embedding
    # ═══════════════════════════════════════════════════════
    async def embed_single_product(
        self, product: Product, *, force: bool = False
    ) -> bool:
        """
        Embed one product.

        Returns True if a new vector was written, False if skipped
        because the source text hasn't changed.
        """
        text = build_embedding_text(product)
        if not text.strip():
            logger.warning(
                "embedding_skipped_empty_text", product_id=str(product.id)
            )
            return False

        text_hash = hash_text(text)

        if not force:
            existing_hash = await self.repo.get_source_hash(
                product.id, self.model
            )
            if existing_hash == text_hash:
                return False

        vector = await self.embedder.embed_one(text)

        await self.repo.upsert(
            product_id=product.id,
            embedding=vector,
            model=self.model,
            source_text=text,
            source_hash=text_hash,
        )
        return True

    # ═══════════════════════════════════════════════════════
    # Batched backfill
    # ═══════════════════════════════════════════════════════
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

        # Build texts, skip empties
        texts: list[str] = []
        product_ids: list[UUID] = []
        for p in products:
            t = build_embedding_text(p)
            if not t.strip():
                stats.failed += 1
                continue
            texts.append(t)
            product_ids.append(p.id)

        if not texts:
            return stats, offset + len(products)

        # Batch embed (embedder handles caching internally)
        try:
            vectors = await self.embedder.embed_batch(texts)
        except Exception as e:
            logger.error("backfill_embed_batch_failed", error=str(e))
            stats.failed += len(product_ids)
            return stats, offset + len(products)

        # Upsert each product in its own savepoint so one failure
        # doesn't abort the entire batch.
        for pid, text, vec in zip(product_ids, texts, vectors):
            savepoint = await self.db.begin_nested()
            try:
                await self.repo.upsert(
                    product_id=pid,
                    embedding=vec,
                    model=self.model,
                    source_text=text,
                    source_hash=hash_text(text),
                )
                await savepoint.commit()
                stats.embedded += 1
            except Exception as e:
                await savepoint.rollback()
                logger.error(
                    "backfill_upsert_failed",
                    product_id=str(pid),
                    error=str(e)[:300],
                )
                stats.failed += 1

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
        """Number of enabled products missing an embedding for this model."""
        return await self.repo.total_missing(self.model)