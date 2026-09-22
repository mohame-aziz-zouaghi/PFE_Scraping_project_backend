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
    on any failure. Idempotent — safe to re-run.
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


@celery_app.task(name="embedding.backfill_missing")
def backfill_missing_task(batch_size: int = 100) -> dict:
    """
    Nightly safety net: find enabled products missing an embedding row
    and embed a batch. Rerun repeatedly to drain the backlog.
    """

    async def _run() -> dict:
        async with async_session_maker() as db:
            svc = EmbeddingBackfillService(db)
            stats, _ = await svc.backfill_batch(
                offset=0, batch_size=batch_size
            )
            return stats.as_dict()

    return asyncio.run(_run())