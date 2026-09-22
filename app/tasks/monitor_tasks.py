"""
Celery tasks wrapping the monitoring engine.

Includes:
- Per-site Redis lock (prevents overlapping cycles)
- Retry only on transient errors
- Structured logging
- Progress tracking via Redis
"""
from celery import shared_task
from redis import Redis
from redis.exceptions import RedisError

from app.config import settings
from app.core.logging_config import get_logger
from app.core.monitoring_adapter import (
    UnavailableMonitoringAdapter,
    get_monitoring_adapter,
)

logger = get_logger(__name__)

# Lock TTL — must exceed the longest possible cycle
LOCK_TTL_SECONDS = 7200


def _get_redis() -> Redis:
    """Sync Redis client for lock management."""
    return Redis.from_url(settings.effective_redis_url, decode_responses=True)


def _acquire_site_lock(site: str) -> bool:
    """Try to acquire an exclusive lock for this site. True = acquired."""
    try:
        r = _get_redis()
        return bool(
            r.set(
                f"monitoring:lock:{site}",
                "1",
                nx=True,
                ex=LOCK_TTL_SECONDS,
            )
        )
    except RedisError as e:
        logger.warning("lock_acquire_failed", site=site, error=str(e))
        # Fail open — allow the task to run rather than block forever
        return True


def _release_site_lock(site: str) -> None:
    try:
        _get_redis().delete(f"monitoring:lock:{site}")
    except RedisError:
        pass


# ═══════════════════════════════════════════════════════
# Tasks
# ═══════════════════════════════════════════════════════

@shared_task(bind=True, max_retries=3)
def run_site_monitoring(self, site: str):
    """
    Run one monitoring cycle for a single site.

    - Acquires an exclusive lock so two cycles for the same site never overlap.
    - Retries on network/transient errors only.
    - Does NOT retry on bad input (unknown site).
    """
    site = (site or "").lower().strip()
    if not site:
        return {"ok": False, "reason": "missing_site"}

    adapter = get_monitoring_adapter()
    if isinstance(adapter, UnavailableMonitoringAdapter):
        logger.error("monitor_task_no_adapter", site=site)
        return {"ok": False, "reason": "no_adapter"}

    # Reject unknown sites early
    known = adapter.enabled_sites()
    if known and site not in known:
        logger.warning("monitor_task_unknown_site", site=site, known=known)
        return {"ok": False, "reason": "unknown_site", "site": site}

    if not _acquire_site_lock(site):
        logger.warning("monitor_task_skipped_locked", site=site)
        return {"ok": False, "reason": "already_running", "site": site}

    try:
        logger.info("monitor_task_start", site=site, task_id=self.request.id)
        adapter.run_site(site)
        logger.info("monitor_task_done", site=site, task_id=self.request.id)
        return {"ok": True, "site": site}

    except (ConnectionError, TimeoutError, RedisError) as e:
        logger.warning("monitor_task_retry", site=site, error=str(e), attempt=self.request.retries + 1)
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))

    except Exception as e:
        logger.error("monitor_task_failed", site=site, error=str(e), exc_info=True)
        return {"ok": False, "reason": "unexpected_error", "error": str(e)}

    finally:
        _release_site_lock(site)


@shared_task
def run_all_sites():
    """Queue monitoring for every enabled site (non-blocking)."""
    adapter = get_monitoring_adapter()
    if isinstance(adapter, UnavailableMonitoringAdapter):
        logger.error("monitor_all_no_adapter")
        return {"queued": 0, "reason": "no_adapter"}

    sites = adapter.enabled_sites()
    for s in sites:
        run_site_monitoring.delay(s)

    logger.info("monitor_all_queued", count=len(sites), sites=sites)
    return {"queued": len(sites), "sites": sites}


@shared_task
def cleanup_old_batches(days: int = 30):
    """
    Delete monitoring_batches older than N days.
    Uses a direct SQLAlchemy connection (sync) to keep this task simple.
    """
    from sqlalchemy import create_engine, text
    from app.config import settings as cfg

    sync_url = cfg.async_database_url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(sync_url)
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    "DELETE FROM monitoring_batches "
                    "WHERE start_time < NOW() - (:days || ' days')::interval"
                ),
                {"days": days},
            )
        logger.info("batch_cleanup_done", deleted=result.rowcount, days=days)
        return {"deleted": result.rowcount}
    finally:
        engine.dispose()