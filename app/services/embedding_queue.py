"""
In-process async queue for background data jobs.

Job kinds:

    ("canonical_link", product_id)
        → link a single product to a canonical + base group.
          Runs the equivalent of backfill_canonical_products
          for one product.

    ("embed_product", product_id)
        → embed a single product into product_embeddings.
          Runs the equivalent of backfill_embeddings for one
          product.

    ("help", article_id)
        → (re)embed a help article.

    ("canonical", canonical_id)
        → (re)embed a canonical product (the canonical_embeddings
          row used by the RAG retriever).

The queue drains by kind: consecutive jobs of the same kind
are batched. A different kind breaks the batch and re-enqueues
the interrupting job for the next iteration.

Logging:
    rag_embedding_queue_started    at startup
    rag_embedding_reconcile        at end of reconciliation
                                   (counts per kind)
    rag_embedding_batch_start      per batch (kind, size)
    rag_embedding_batch_done       per batch (kind, size, ms)
    rag_embedding_progress         every PROGRESS_LOG_EVERY jobs
                                   (processed, queue_size, kind)
    rag_embedding_drained          when queue empties after work
    rag_embedding_idle             when a worker idles
    rag_embedding_resumed          when a worker wakes up
"""
from __future__ import annotations

import asyncio
import time
from typing import Literal, Optional
from uuid import UUID

from app.core.logging_config import get_logger

logger = get_logger(__name__)

JobKind = Literal[
    "canonical_link", "embed_product", "help", "canonical"
]


# ═══════════════════════════════════════════════════════
# Tunables
# ═══════════════════════════════════════════════════════
WORKER_COUNT = 2
BATCH_SIZE = 64
PROGRESS_LOG_EVERY = 200        # log every N jobs processed
IDLE_TIMEOUT_S = 1.0


class EmbeddingQueue:
    def __init__(self) -> None:
        self._queue: Optional[asyncio.Queue[tuple[JobKind, UUID]]] = None
        self._workers: list[asyncio.Task] = []
        self._stop = asyncio.Event()

        # Counters (safe in single event loop; no locking needed).
        self._processed_since_start = 0
        self._last_log_at = 0
        # Per-kind counters (incremented in _run_worker).
        self._processed_by_kind: dict[JobKind, int] = {
            "canonical_link": 0,
            "embed_product": 0,
            "help": 0,
            "canonical": 0,
        }

    # ── Lifecycle ──────────────────────────────────────
    def start(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._workers and not all(w.done() for w in self._workers):
            return
        self._stop.clear()
        self._workers = [
            asyncio.create_task(
                self._run_worker(idx),
                name=f"rag_embedding_worker_{idx}",
            )
            for idx in range(WORKER_COUNT)
        ]
        logger.info(
            "rag_embedding_queue_started",
            workers=WORKER_COUNT,
            batch_size=BATCH_SIZE,
        )

    async def stop(self) -> None:
        self._stop.set()
        for w in self._workers:
            if not w.done():
                try:
                    await asyncio.wait_for(w, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("rag_embedding_worker_stop_timeout")
                    w.cancel()

    # ── Public API ─────────────────────────────────────
    def enqueue(self, kind: JobKind, id_: UUID) -> None:
        if self._queue is None:
            logger.debug(
                "rag_embedding_enqueue_before_start",
                kind=kind, id=str(id_),
            )
            return
        self._queue.put_nowait((kind, id_))

    # ── Worker ─────────────────────────────────────────
    async def _run_worker(self, worker_idx: int) -> None:
        assert self._queue is not None
        was_idle = False
        idle_started_at = 0.0

        while not self._stop.is_set():
            try:
                kind, id_ = await asyncio.wait_for(
                    self._queue.get(), timeout=IDLE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                if not was_idle:
                    was_idle = True
                    idle_started_at = time.monotonic()
                    logger.info(
                        "rag_embedding_idle",
                        worker=worker_idx,
                        processed_total=self._processed_since_start,
                    )
                continue
            except asyncio.CancelledError:
                break

            if was_idle:
                was_idle = False
                logger.info(
                    "rag_embedding_resumed",
                    worker=worker_idx,
                    idle_seconds=int(
                        time.monotonic() - idle_started_at
                    ),
                )

            # Drain a batch of same-kind jobs.
            batch = await self._drain_batch(
                kind=kind, first_id=id_, max_size=BATCH_SIZE
            )

            batch_started = time.monotonic()
            succeeded = False
            try:
                logger.info(
                    "rag_embedding_batch_start",
                    kind=kind,
                    size=len(batch),
                    worker=worker_idx,
                )
                await self._process_batch(kind=kind, ids=batch)
                succeeded = True
            except Exception as exc:
                logger.warning(
                    "rag_embedding_batch_failed",
                    kind=kind,
                    size=len(batch),
                    error=str(exc)[:200],
                )
            finally:
                for _ in batch:
                    self._queue.task_done()

                batch_ms = int(
                    (time.monotonic() - batch_started) * 1000
                )

                # Counters only advance on success — a failed
                # batch did not actually process its rows.
                if succeeded:
                    self._processed_since_start += len(batch)
                    self._processed_by_kind[kind] = (
                        self._processed_by_kind.get(kind, 0)
                        + len(batch)
                    )

                queue_now = self._queue.qsize()

                logger.info(
                    "rag_embedding_batch_done",
                    kind=kind,
                    size=len(batch),
                    ms=batch_ms,
                    queue_size=queue_now,
                    worker=worker_idx,
                    succeeded=succeeded,
                )

                # Periodic summary.
                if (
                    self._processed_since_start - self._last_log_at
                    >= PROGRESS_LOG_EVERY
                ):
                    self._last_log_at = self._processed_since_start
                    logger.info(
                        "rag_embedding_progress",
                        processed=self._processed_since_start,
                        queue_size=queue_now,
                        **{
                            f"done_{k}": v
                            for k, v in self._processed_by_kind.items()
                        },
                    )

                if queue_now == 0:
                    logger.info(
                        "rag_embedding_drained",
                        processed_total=self._processed_since_start,
                        **{
                            f"done_{k}": v
                            for k, v in self._processed_by_kind.items()
                        },
                    )

    async def _drain_batch(
        self,
        *,
        kind: JobKind,
        first_id: UUID,
        max_size: int,
    ) -> list[UUID]:
        assert self._queue is not None
        batch: list[UUID] = [first_id]
        seen: set[UUID] = {first_id}

        while len(batch) < max_size:
            try:
                k, i = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if k != kind:
                self._queue.put_nowait((k, i))
                break

            if i in seen:
                self._queue.task_done()
                continue

            seen.add(i)
            batch.append(i)

        return batch

    async def _process_batch(
        self, *, kind: JobKind, ids: list[UUID]
    ) -> None:
        if not ids:
            return

        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            if kind == "canonical_link":
                from app.services.canonical_linker import CanonicalLinker
                linker = CanonicalLinker(session)
                await linker.link_products(ids)

            elif kind == "embed_product":
                from app.services.embedding_service import EmbeddingService
                service = EmbeddingService(db=session)
                await service.embed_products_bulk(ids)

            elif kind == "help":
                from app.services.embedding_service import EmbeddingService
                service = EmbeddingService(db=session)
                await service.embed_help_articles_bulk(ids)

            elif kind == "canonical":
                from app.services.embedding_service import EmbeddingService
                service = EmbeddingService(db=session)
                await service.embed_canonical_products_bulk(ids)

    # ── Reconciliation ─────────────────────────────────
    async def reconcile_all(self) -> dict[str, int]:
        """
        Called from startup. Finds every row that needs
        processing, enqueues jobs, and logs a full snapshot
        before returning.

        Counts returned:

            canonical_link   products with canonical_id IS NULL
            embed_product    products without a product_embeddings row
            canonical        canonicals without a canonical_embeddings row
            help             articles without a help_embeddings row

        All queries use index-friendly LEFT JOIN ... IS NULL
        patterns and run in parallel where it's safe to do so.
        """
        from app.db.session import AsyncSessionLocal
        from sqlalchemy import text

        t0 = time.monotonic()
        result = {
            "canonical_link": 0,
            "embed_product": 0,
            "help": 0,
            "canonical": 0,
        }

        async with AsyncSessionLocal() as session:
            # Snapshot the pending counts first so the operator
            # sees the total work before any batches run.
            snap = (await session.execute(text(
                """
                SELECT
                  (SELECT COUNT(*) FROM products p
                   WHERE p.is_enabled = TRUE
                     AND p.canonical_id IS NULL)
                    AS pending_canonical_link,
                  (SELECT COUNT(*) FROM products p
                   LEFT JOIN product_embeddings pe
                     ON pe.product_id = p.id
                   WHERE p.is_enabled = TRUE
                     AND pe.id IS NULL)
                    AS pending_embed_product,
                  (SELECT COUNT(*) FROM canonical_products cp
                   LEFT JOIN canonical_embeddings ce
                     ON ce.canonical_id = cp.id
                   WHERE cp.listing_count > 0
                     AND ce.id IS NULL)
                    AS pending_canonical,
                  (SELECT COUNT(*) FROM help_articles ha
                   LEFT JOIN help_embeddings he
                     ON he.article_id = ha.id
                   WHERE ha.is_published = TRUE
                     AND he.id IS NULL)
                    AS pending_help
                """
            ))).one()

            logger.info(
                "rag_reconcile_snapshot",
                pending_canonical_link=snap.pending_canonical_link,
                pending_embed_product=snap.pending_embed_product,
                pending_canonical=snap.pending_canonical,
                pending_help=snap.pending_help,
            )

            # ── 1. Products needing canonical linkage
            if snap.pending_canonical_link > 0:
                rows = (await session.execute(text(
                    """
                    SELECT p.id
                    FROM products p
                    WHERE p.is_enabled = TRUE
                      AND p.canonical_id IS NULL
                    """
                ))).all()
                for (pid,) in rows:
                    self.enqueue("canonical_link", pid)
                result["canonical_link"] = len(rows)

            # ── 2. Products needing embeddings
            if snap.pending_embed_product > 0:
                rows = (await session.execute(text(
                    """
                    SELECT p.id
                    FROM products p
                    LEFT JOIN product_embeddings pe
                      ON pe.product_id = p.id
                    WHERE p.is_enabled = TRUE
                      AND pe.id IS NULL
                    """
                ))).all()
                for (pid,) in rows:
                    self.enqueue("embed_product", pid)
                result["embed_product"] = len(rows)

            # ── 3. Canonicals needing embeddings
            if snap.pending_canonical > 0:
                rows = (await session.execute(text(
                    """
                    SELECT cp.id
                    FROM canonical_products cp
                    LEFT JOIN canonical_embeddings ce
                      ON ce.canonical_id = cp.id
                    WHERE cp.listing_count > 0
                      AND ce.id IS NULL
                    """
                ))).all()
                for (cid,) in rows:
                    self.enqueue("canonical", cid)
                result["canonical"] = len(rows)

            # ── 4. Help articles needing embeddings
            if snap.pending_help > 0:
                rows = (await session.execute(text(
                    """
                    SELECT ha.id
                    FROM help_articles ha
                    LEFT JOIN help_embeddings he
                      ON he.article_id = ha.id
                    WHERE ha.is_published = TRUE
                      AND he.id IS NULL
                    """
                ))).all()
                for (aid,) in rows:
                    self.enqueue("help", aid)
                result["help"] = len(rows)

        total_queued = sum(result.values())
        snapshot_s = time.monotonic() - t0

        # Estimate completion: batch_size jobs per iteration,
        # 2 workers in parallel, ~700 ms per Ollama call per
        # batch. Only the embedding kinds pay the 700 ms cost;
        # canonical_link is DB-only.
        embedding_jobs = (
            result["embed_product"]
            + result["canonical"]
            + result["help"]
        )
        batches = max(1, (embedding_jobs + BATCH_SIZE - 1) // BATCH_SIZE)
        est_seconds = int(
            (batches * 700) / max(1, WORKER_COUNT) / 1000
        )

        if total_queued > 0:
            logger.info(
                "rag_embedding_reconcile",
                **result,
                total_queued=total_queued,
                queue_size=self._queue.qsize() if self._queue else 0,
                snapshot_ms=int(snapshot_s * 1000),
                estimated_completion_s=est_seconds,
            )
        else:
            logger.info(
                "rag_embedding_reconcile",
                total_queued=0,
                snapshot_ms=int(snapshot_s * 1000),
            )

        return result


# Module-level singleton
embedding_queue = EmbeddingQueue()