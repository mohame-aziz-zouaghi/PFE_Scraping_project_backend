"""
Notification boundary.

Applications can plug in a real sender (email, Slack, webhook) later.
By default, notifications are logged but not delivered.
"""
from typing import Any

from app.core.logging_config import get_logger
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(
    name="app.tasks.notification_tasks.deliver_alert",
    bind=True,
    max_retries=3,
)
def deliver_alert(self, alert: dict[str, Any]) -> dict[str, Any]:
    """
    Accept a structured alert dict.

    Real delivery is NOT implemented — returns a no-op response.
    Replace the body with your actual sender when ready:

        from app.core.notifications import send_email
        send_email(to=..., subject=..., body=...)
    """
    logger.info(
        "notification_not_configured",
        alert_id=alert.get("id"),
        site=alert.get("site"),
        alert_type=alert.get("alert_type"),
        severity=alert.get("severity"),
    )
    return {
        "delivered": False,
        "reason": "notification adapter is not configured",
        "alert_id": alert.get("id"),
    }


@celery_app.task(
    name="app.tasks.notification_tasks.cleanup_old_alerts",
    bind=True,
)
def cleanup_old_alerts(self, days: int = 90) -> dict[str, Any]:
    """
    Auto-resolve alerts older than N days to keep the alerts table small.
    """
    from sqlalchemy import create_engine, text
    from app.config import settings as cfg

    sync_url = cfg.async_database_url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(sync_url)
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    "UPDATE monitoring_alerts "
                    "SET is_resolved = TRUE, resolved_at = NOW() "
                    "WHERE is_resolved = FALSE "
                    "AND created_at < NOW() - (:days || ' days')::interval"
                ),
                {"days": days},
            )
        logger.info("alerts_cleanup_done", resolved=result.rowcount, days=days)
        return {"resolved": result.rowcount}
    finally:
        engine.dispose()