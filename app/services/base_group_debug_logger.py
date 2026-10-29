# app/services/base_group_debug_logger.py
"""
Debug logger for base_group changes.

When settings.base_group_debug_log is true, this logger:

  1. Emits a structured log line per mutation.
  2. Writes a row to base_group_change_log.

When false, every method is a no-op (except a single check
of the setting, so the runtime cost is negligible).

The toggle is read from settings at construction time. If you
change the env var at runtime, the worker must restart to
pick it up — but the audit table is harmless to leave populated
even after the toggle is off.
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class BaseGroupDebugLogger:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.enabled = bool(
            getattr(settings, "base_group_debug_log", False)
        )

    async def log(
        self,
        *,
        base_group_id: UUID,
        canonical_id: Optional[UUID],
        change_type: str,               # 'created' | 'updated' | 'deleted'
        trigger_source: str,            # 'targeted_rebuild' | 'full_match' | ...
        product_ids: list[UUID],
        details: Optional[dict] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Record one base_group mutation."""
        if not self.enabled:
            return

        # Structured log line
        logger.info(
            "base_group_change",
            base_group_id=str(base_group_id),
            canonical_id=(
                str(canonical_id) if canonical_id else None
            ),
            change_type=change_type,
            trigger_source=trigger_source,
            product_count=len(product_ids),
            product_ids=[str(x) for x in product_ids[:20]],
            task_id=task_id,
            **(details or {}),
        )

        # Audit table
        try:
            await self.db.execute(
                text("""
                    INSERT INTO base_group_change_log (
                        base_group_id, canonical_id, change_type,
                        trigger_source, product_ids, details, task_id
                    ) VALUES (
                        :bg, :cid, :ct, :src,
                        CAST(:pids AS uuid[]),
                        CAST(:details AS jsonb),
                        :task
                    )
                """),
                {
                    "bg": str(base_group_id),
                    "cid": str(canonical_id) if canonical_id else None,
                    "ct": change_type,
                    "src": trigger_source,
                    "pids": [str(x) for x in product_ids],
                    "details": (
                        __import__("json").dumps(details)
                        if details else None
                    ),
                    "task": task_id,
                },
            )
        except Exception as exc:
            # Never let debug logging crash the pipeline
            logger.warning(
                "base_group_change_log_failed",
                base_group_id=str(base_group_id),
                error=str(exc)[:200],
            )

    async def log_batch(
        self,
        *,
        changes: list[dict],
        trigger_source: str,
        task_id: Optional[str] = None,
    ) -> None:
        """
        Record multiple base_group changes in one call.

        `changes` is a list of dicts, each with:
            base_group_id  (UUID)
            canonical_id   (UUID | None)
            change_type    (str)
            product_ids    (list[UUID])
            details        (dict | None, optional)
        """
        if not self.enabled or not changes:
            return

        for change in changes:
            await self.log(
                base_group_id=change["base_group_id"],
                canonical_id=change.get("canonical_id"),
                change_type=change["change_type"],
                trigger_source=trigger_source,
                product_ids=change.get("product_ids") or [],
                details=change.get("details"),
                task_id=task_id,
            )