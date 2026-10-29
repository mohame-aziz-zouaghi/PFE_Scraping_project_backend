# app/db/dynamic_pool.py
"""
Dynamic async pool: grows min -> max on demand, evicts idle > timeout.

FIX (this revision):
    The prior version navigated `self._pool.mutex` / `self._pool.queue`,
    which do not exist on SQLAlchemy 2.0's AsyncAdaptedQueue. The reaper
    now reads `self._available_connections` directly — a plain Python
    list that QueuePool keeps as an instance attribute. No mutex, no
    greenlet bridge, no lock.

Config (env-driven, read at construction):
    POOL_MIN_SIZE           default 5
    POOL_MAX_SIZE           default 30
    POOL_IDLE_TIMEOUT_S     default 5
    POOL_REAPER_INTERVAL_S  default 1
    POOL_ACQUIRE_TIMEOUT_S  default 10
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy.pool.base import ConnectionPoolEntry

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


class DynamicAsyncPool(AsyncAdaptedQueuePool):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        min_size = _env_int("POOL_MIN_SIZE", 5)
        max_size = _env_int("POOL_MAX_SIZE", 30)
        idle_timeout = _env_float("POOL_IDLE_TIMEOUT_S", 5.0)
        reaper_interval = _env_float("POOL_REAPER_INTERVAL_S", 1.0)
        acquire_timeout = _env_float("POOL_ACQUIRE_TIMEOUT_S", 10.0)

        if min_size > max_size:
            raise ValueError(
                f"POOL_MIN_SIZE ({min_size}) > POOL_MAX_SIZE ({max_size})"
            )
        if min_size < 1:
            raise ValueError("POOL_MIN_SIZE must be >= 1")

        kwargs.pop("pool_size", None)
        kwargs.pop("max_overflow", None)
        kwargs.pop("timeout", None)

        super().__init__(
            *args,
            pool_size=min_size,
            max_overflow=(max_size - min_size),
            timeout=acquire_timeout,
            **kwargs,
        )

        self._min_size = min_size
        self._max_size = max_size
        self._idle_timeout = idle_timeout
        self._reaper_interval = reaper_interval

        # rec_id (id(connection_record)) -> monotonic timestamp
        self._last_used_at: dict[int, float] = {}
        self._reaper_task: asyncio.Task | None = None

        logger.info(
            "dynamic_pool_created | min=%d max=%d idle_timeout=%.1fs",
            min_size, max_size, idle_timeout,
        )

    # ── Tracking hooks ────────────────────────────────────────────────
    def _do_return_conn(self, conn: ConnectionPoolEntry) -> None:
        try:
            super()._do_return_conn(conn)
        finally:
            self._last_used_at[id(conn)] = time.monotonic()

    def _do_get(self) -> ConnectionPoolEntry:
        entry = super()._do_get()
        self._last_used_at.pop(id(entry), None)
        return entry

    # ── Reaper ────────────────────────────────────────────────────────
    def start_reaper(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._reaper_task is not None and not self._reaper_task.done():
            return
        try:
            self._reaper_task = loop.create_task(
                self._reap_loop(), name="db_pool_reaper"
            )
            logger.info("dynamic_pool_reaper_started")
        except RuntimeError:
            # Loop already closed (shutdown race)
            pass

    async def stop_reaper(self) -> None:
        if self._reaper_task is None:
            return
        self._reaper_task.cancel()
        try:
            await self._reaper_task
        except (asyncio.CancelledError, Exception):
            pass
        self._reaper_task = None
        logger.info("dynamic_pool_reaper_stopped")

    async def _reap_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._reaper_interval)
                try:
                    self._reap_idle()
                except Exception as exc:
                    logger.warning(
                        "dynamic_pool_reap_error | %r", exc
                    )
        except asyncio.CancelledError:
            pass

    def _reap_idle(self) -> None:
        """
        Close connections idle >= idle_timeout, down to min_size.

        Reads `_available_connections` directly — the plain list that
        QueuePool keeps on the instance. No mutex, no greenlet bridge.
        Safe because the reaper runs on the same event loop the pool
        uses, so no two operations touch the list concurrently.
        """
        available = getattr(self, "_available_connections", None)
        if not available:
            return

        now = time.monotonic()

        # Size = checked_out + available
        checked_out = self.checkedout()
        current_size = checked_out + len(available)
        evictable = current_size - self._min_size
        if evictable <= 0:
            return

        # Oldest-first snapshot — sorted by last_used_at (0.0 if unknown)
        snapshot = sorted(
            available,
            key=lambda rec: self._last_used_at.get(id(rec), 0.0),
        )

        evicted = 0
        for rec in snapshot:
            if evicted >= evictable:
                break
            rec_id = id(rec)
            last_used = self._last_used_at.get(rec_id, 0.0)
            if (now - last_used) < self._idle_timeout:
                # Sorted — all remaining are younger
                break

            # Remove from the pool's idle list before closing so a
            # concurrent checkout can't grab a connection we're
            # about to close.
            try:
                available.remove(rec)
            except ValueError:
                # Already gone (checkout raced) — just drop tracking
                self._last_used_at.pop(rec_id, None)
                continue

            try:
                rec.close()
                evicted += 1
            except Exception as exc:
                logger.warning(
                    "dynamic_pool_evict_failed | %r", exc
                )
            finally:
                self._last_used_at.pop(rec_id, None)

        if evicted:
            logger.debug(
                "dynamic_pool_evicted | count=%d size_now=%d",
                evicted, self.checkedout() + len(self._available_connections),
            )

    # ── Introspection ─────────────────────────────────────────────────
    def dynamic_stats(self) -> dict:
        available = getattr(self, "_available_connections", [])
        return {
            "size":           self.checkedout() + len(available),
            "checked_out":    self.checkedout(),
            "idle":           len(available),
            "min_size":       self._min_size,
            "max_size":       self._max_size,
            "idle_timeout_s": self._idle_timeout,
        }