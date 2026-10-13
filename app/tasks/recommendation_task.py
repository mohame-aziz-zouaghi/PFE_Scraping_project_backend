"""Celery tasks for the recommendation engine."""
from __future__ import annotations

import asyncio

from app.core.logging_config import get_logger
from app.core.recommendation_config import MATERIALIZATION
from app.db.session import async_session_maker
from app.repositories.profile_repo import ProfileRepo
from app.repositories.recommendation_repo import RecommendationRepo
from app.services.profile_service import ProfileService
from app.services.recommendation_service import RecommendationService
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="recs.refresh_all_users")
def refresh_all_users() -> dict:
    """
    Nightly refresh.

    Two passes:
      1. Rebuild every expired user profile from the last
         PROFILE.EVENT_WINDOW_DAYS of events.
      2. Regenerate the materialized recommendations for those
         users so the next /for-me request hits a fresh cache.

    Bounded by MATERIALIZATION.MAX_USERS_PER_RUN.
    """
    return asyncio.run(_refresh_all_users_async())


async def _refresh_all_users_async() -> dict:
    rebuilt = 0
    regenerated = 0
    errors = 0

    async with async_session_maker() as db:
        profile_repo = ProfileRepo(db)
        rec_repo = RecommendationRepo(db)
        profile_svc = ProfileService(db)
        rec_svc = RecommendationService(db)

        # ── Pass 1: rebuild expired profiles.
        expired_ids = await profile_repo.list_expired_user_ids(
            limit=MATERIALIZATION.MAX_USERS_PER_RUN
        )

        for uid in expired_ids:
            try:
                await profile_svc.rebuild(uid)
                rebuilt += 1
            except Exception as exc:
                errors += 1
                logger.warning(
                    "profile_rebuild_failed",
                    user_id=str(uid),
                    error=str(exc)[:200],
                )

        # ── Pass 2: regenerate materialized recommendations
        #    for the same users (their profile is now fresh).
        for uid in expired_ids:
            try:
                await rec_svc.for_user(user_id=uid)
                regenerated += 1
            except Exception as exc:
                errors += 1
                logger.warning(
                    "recommendation_refresh_failed",
                    user_id=str(uid),
                    error=str(exc)[:200],
                )

    logger.info(
        "recs_refresh_complete",
        profiles_rebuilt=rebuilt,
        recommendations_regenerated=regenerated,
        errors=errors,
    )

    return {
        "profiles_rebuilt": rebuilt,
        "recommendations_regenerated": regenerated,
        "errors": errors,
    }


@celery_app.task(name="recs.purge_expired")
def purge_expired() -> dict:
    """Delete expired materialized recommendation rows."""
    return asyncio.run(_purge_expired_async())


async def _purge_expired_async() -> dict:
    async with async_session_maker() as db:
        repo = RecommendationRepo(db)
        deleted = await repo.purge_expired()
        await db.commit()
    logger.info("recs_purge_complete", deleted=deleted)
    return {"deleted": deleted}