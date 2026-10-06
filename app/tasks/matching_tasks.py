"""Celery tasks for the matching pipeline."""
from __future__ import annotations

import asyncio
from typing import Optional

from celery import shared_task

from app.tasks.celery_app import celery_app


# ═══════════════════════════════════════════════════════
# SAFE — the only one the scheduler should call
# ═══════════════════════════════════════════════════════
@celery_app.task(
    name="matching.nightly",
    bind=True,
    time_limit=60 * 60 * 12,
    soft_time_limit=60 * 60 * 11,
)
def nightly(self) -> dict:
    """Nightly incremental pass. Never destructive. Never LLM-budgeted
    to death — uses the config default."""
    return _run_match_sync(
        max_distance=0.25,
        max_candidates=None,
        llm_budget=None,
        clear_existing=False,
    )


# ═══════════════════════════════════════════════════════
# DESTRUCTIVE — only callable via API with full confirmation
# ═══════════════════════════════════════════════════════
@celery_app.task(
    name="matching.full_run",
    bind=True,
    time_limit=60 * 60 * 12,
    soft_time_limit=60 * 60 * 11,
)
def full_run(
    self,
    *,
    max_distance: float = 0.25,
    max_candidates: Optional[int] = None,
    llm_budget: Optional[int] = None,
    clear_existing: bool = False,
    requested_by: Optional[str] = None,
) -> dict:
    """Full rebuild. Only enqueued by the superadmin API path.

    Additional safety: refuses clear_existing=true if it was scheduled
    automatically (i.e. no requested_by user attached).
    """
    if clear_existing and not requested_by:
        raise ValueError(
            "clear_existing=true is forbidden on auto-scheduled tasks. "
            "Only a superadmin API call may pass it."
        )
    return _run_match_sync(
        max_distance=max_distance,
        max_candidates=max_candidates,
        llm_budget=llm_budget,
        clear_existing=clear_existing,
    )


def _run_match_sync(
    *,
    max_distance: float,
    max_candidates: Optional[int],
    llm_budget: Optional[int],
    clear_existing: bool,
) -> dict:
    from app.db.redis_client import close_redis
    from app.db.session import async_session_maker, dispose_engine
    from app.services.matching_service import MatchingService

    async def _run() -> dict:
        try:
            async with async_session_maker() as db:
                svc = MatchingService(db)
                stats = await svc.run_full_match(
                    max_distance=max_distance,
                    max_candidates=max_candidates,
                    llm_budget=llm_budget,
                    clear_existing=clear_existing,
                )
                return stats.as_dict()
        finally:
            try:
                await close_redis()
            except Exception:
                pass
            try:
                await dispose_engine()
            except Exception:
                pass

    return asyncio.run(_run())