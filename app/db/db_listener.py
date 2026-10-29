# app/db/pg_listener.py
"""
PostgreSQL LISTEN connection.

Channels:
    product_inserted  — new products from the scraper → embedding queue
    alert_created     — new monitoring alerts → alert fan-out
"""
from __future__ import annotations

import asyncio
from typing import Optional
from uuid import UUID

import asyncpg

from app.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class PgListener:
    def __init__(self) -> None:
        self._conn: Optional[asyncpg.Connection] = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="pg_listener")
        logger.info(
            "pg_listener_started",
            channels=["product_inserted", "alert_created"],
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def _run(self) -> None:
        backoff_s = 1.0
        while not self._stop.is_set():
            try:
                await self._connect_and_listen()
                backoff_s = 1.0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "pg_listener_connection_failed",
                    error=str(exc)[:200],
                    retry_in=int(backoff_s),
                )
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=backoff_s
                    )
                except asyncio.TimeoutError:
                    pass
                backoff_s = min(60.0, backoff_s * 2)

    async def _connect_and_listen(self) -> None:
        dsn = settings.async_database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )

        self._conn = await asyncpg.connect(dsn)
        await self._conn.add_listener(
            "product_inserted", self._on_product_inserted
        )
        await self._conn.add_listener(
            "alert_created", self._on_alert_created
        )
        logger.info(
            "pg_listener_connected",
            channels=["product_inserted", "alert_created"],
        )

        try:
            await self._stop.wait()
        finally:
            if self._conn is not None:
                try:
                    await self._conn.remove_listener(
                        "product_inserted",
                        self._on_product_inserted,
                    )
                except Exception:
                    pass
                try:
                    await self._conn.remove_listener(
                        "alert_created", self._on_alert_created,
                    )
                except Exception:
                    pass
                try:
                    await self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def _on_product_inserted(
        self,
        connection: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        if self._stop.is_set():
            return
        try:
            product_id = UUID(payload)
        except (ValueError, TypeError):
            logger.warning(
                "pg_listener_bad_payload",
                channel=channel, payload=payload[:64],
            )
            return
        from app.services.embedding_queue import embedding_queue
        embedding_queue.enqueue("canonical_link", product_id)
        embedding_queue.enqueue("embed_product", product_id)

    def _on_alert_created(
        self,
        connection: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        if self._stop.is_set():
            return
        import json
        try:
            data = json.loads(payload)
            alert_id    = int(data["alert_id"])
            product_id  = UUID(data["product_id"])
            site        = str(data.get("site", ""))
            alert_type  = str(data.get("alert_type", ""))
            severity    = str(data.get("severity", "medium"))
            old_value   = data.get("old_value")
            new_value   = data.get("new_value")
            change_pct  = data.get("change_percent")
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "pg_listener_bad_alert_payload",
                payload=payload[:200],
                error=str(exc)[:100],
            )
            return

        from app.services.alert_fanout import AlertEvent, alert_fanout
        alert_fanout.enqueue(AlertEvent(
            alert_id=alert_id,
            product_id=product_id,
            site=site,
            alert_type=alert_type,
            severity=severity,
            old_value=old_value,
            new_value=new_value,
            change_pct=change_pct,
        ))


pg_listener = PgListener()