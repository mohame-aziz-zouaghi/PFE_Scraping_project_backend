# app/services/alert_fanout.py
"""
Alert → user notification fan-out.

Consumes AlertEvent, calls resolve_targets (SQL), sorts by
online/offline, applies rate limits, dispatches.

Two delivery paths:

    ONLINE   — immediate WS push + immediate email (if opted).
               Rate-limited: max 2/1h. If exceeded → hand off to
               the offline digest path so it isn't dropped.

    OFFLINE  — email digest. Rate-limited: max 2/6h. If exceeded,
               alert stays unsent; sweep task replays it in next
               window.

Two Celery task families called from here:
    notifications.send_user_alert_digest    (offline + overflow online)
    notifications.send_user_alert_immediate (online per-alert push+email)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from app.core.logging_config import get_logger
from app.tasks.notification_tasks import send_user_alert_digest, send_user_alert_immediate

logger = get_logger(__name__)


QUEUE_MAXSIZE       = 5000
RESOLVER_COUNT      = 1
SENDER_COUNT        = 8
RESOLVER_BATCH_MAX  = 50


@dataclass(slots=True)
class AlertEvent:
    alert_id:    int
    product_id:  UUID
    site:        str
    alert_type:  str
    severity:    str
    old_value:   Optional[str] = None
    new_value:   Optional[str] = None
    change_pct:  Optional[float] = None
    enqueued_at: float = field(default_factory=time.monotonic)


class AlertFanout:
    def __init__(self) -> None:
        self._queue: Optional[asyncio.Queue[AlertEvent]] = None
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._dropped = 0
        self._processed = 0

    # ── Lifecycle ──────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("alert_fanout_start_no_loop")
            return
        self._queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._running = True
        for i in range(RESOLVER_COUNT):
            self._workers.append(loop.create_task(
                self._resolver_loop(i),
                name=f"alert_fanout_resolver_{i}",
            ))
        logger.info(
            "alert_fanout_started",
            resolvers=RESOLVER_COUNT,
            queue_max=QUEUE_MAXSIZE,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._queue is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "alert_fanout_drain_timeout",
                    remaining=self._queue.qsize(),
                )
        for t in self._workers:
            t.cancel()
        for t in self._workers:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._workers.clear()
        logger.info(
            "alert_fanout_stopped",
            processed=self._processed,
            dropped=self._dropped,
        )

    def enqueue(self, event: AlertEvent) -> None:
        if not self._running or self._queue is None:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                pass
            try:
                self._queue.put_nowait(event)
            except Exception:
                pass
            if self._dropped % 100 == 1:
                logger.warning(
                    "alert_fanout_queue_full", dropped=self._dropped,
                )

    def stats(self) -> dict:
        return {
            "running":   self._running,
            "queued":    self._queue.qsize() if self._queue else 0,
            "processed": self._processed,
            "dropped":   self._dropped,
            "workers":   len(self._workers),
        }

    # ── Resolver loop ──────────────────────────────────
    async def _resolver_loop(self, worker_id: int) -> None:
        from app.db import acquire_db
        from app.services.alert_rules import resolve_targets
        from app.services.alert_rate_limit import (
            try_consume_online,
            try_consume_offline,
        )

        while self._running:
            # Gather a batch
            batch: list[AlertEvent] = []
            try:
                batch.append(await self._queue.get())  # type: ignore[union-attr]
                for _ in range(RESOLVER_BATCH_MAX - 1):
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    "alert_resolver_dequeue_error",
                    error=str(exc)[:200],
                )
                await asyncio.sleep(1.0)
                continue

            try:
                alert_ids = [a.alert_id for a in batch]
                async with acquire_db(readonly=True) as db:
                    targets = await resolve_targets(db, alert_ids=alert_ids)

                # Track which (user, alert) pairs we've handled so
                # repeated pairs (many alerts same product) collapse.
                seen: set[tuple[str, int]] = set()

                for t in targets:
                    key = (t.user_id, t.alert_id)
                    if key in seen:
                        continue
                    seen.add(key)

                    if t.online:
                        allowed, _ = await try_consume_online(t.user_id)
                        if allowed:
                            await self._dispatch_online(
                                t.user_id, t.alert_id,
                            )
                        else:
                            # Rate-limited online → overflow to
                            # digest (offline path) so we don't
                            # lose the notification.
                            allowed_off, _ = await try_consume_offline(
                                t.user_id,
                            )
                            if allowed_off:
                                self._queue_digest(
                                    t.user_id, [t.alert_id],
                                )
                            else:
                                # Offline ceiling also hit — leave
                                # unsent; sweep task will replay
                                # in the next window.
                                logger.info(
                                    "alert_rate_limited_both_windows",
                                    user_id=t.user_id,
                                    alert_id=t.alert_id,
                                )
                    else:
                        allowed, _ = await try_consume_offline(t.user_id)
                        if allowed:
                            self._queue_digest(t.user_id, [t.alert_id])
                        else:
                            logger.info(
                                "alert_rate_limited_offline",
                                user_id=t.user_id,
                                alert_id=t.alert_id,
                            )
                        # else: leave unsent, sweep retries in 6h

                self._processed += len(batch)
            except Exception as exc:
                logger.warning(
                    "alert_resolve_batch_failed",
                    batch_size=len(batch),
                    error=str(exc)[:300],
                )
            finally:
                for _ in batch:
                    try:
                        self._queue.task_done()  # type: ignore[union-attr]
                    except Exception:
                        pass

    # ── Dispatch helpers ───────────────────────────────
    async def _dispatch_online(self, user_id: str, alert_id: int) -> None:
        """
        Online: WS push now + Celery immediate task for email + DB row.
        The immediate task handles the DB insert and email so the
        resolver loop never blocks on I/O.
        """
        # WS push is fast, do it inline for latency
        try:
            from app.api.v1.websocket import push_alert_to_user
            await push_alert_to_user(user_id, alert_id)
        except Exception as exc:
            logger.debug(
                "alert_ws_push_skipped",
                user_id=user_id,
                alert_id=alert_id,
                error=str(exc)[:200],
            )

        # Email + DB inbox via Celery (never blocks)
        try:
            send_user_alert_immediate.delay(
                user_id=user_id, alert_id=alert_id,
            )
        except Exception as exc:
            logger.warning(
                "alert_immediate_enqueue_failed",
                user_id=user_id,
                alert_id=alert_id,
                error=str(exc)[:200],
            )

    def _queue_digest(self, user_id: str, alert_ids: list[int]) -> None:
        try:
            send_user_alert_digest.delay(
                user_id=user_id,
                alert_ids=alert_ids,
                window_seconds=60,
            )
        except Exception as exc:
            logger.warning(
                "alert_digest_enqueue_failed",
                user_id=user_id,
                error=str(exc)[:200],
            )


alert_fanout = AlertFanout()