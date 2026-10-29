"""
Celery application configuration.

Uses separate Redis DBs for broker and result backend to prevent
cache eviction from wiping task results.

    DB 0 -> application cache (progress, sessions, rate limits)
    DB 1 -> Celery broker queue
    DB 2 -> Celery task results
"""
from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "monitoring",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    # Every module that defines @shared_task / @task must be listed
    # here. Celery imports these at worker startup; if a module is
    # missing from this list, its tasks are not registered and any
    # message that targets them is discarded with:
    #   "Received unregistered task of type 'X'"
    include=[
        "app.tasks.monitor_tasks",
        "app.tasks.notification_tasks",
        "app.tasks.matching_tasks",
        "app.tasks.embedding_tasks",
        "app.tasks.recommendation_tasks",
        "app.tasks.verification_tasks",
    ],
)

celery_app.conf.update(
    # ─── Serialization ────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ─── Time ─────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,

    # ─── Task tracking ────────────────────────────────
    task_track_started=True,
    task_send_sent_event=True,
    task_acks_late=True,

    # ─── Limits ───────────────────────────────────────
    task_time_limit=3600,
    task_soft_time_limit=3300,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=200,

    # ─── Results ──────────────────────────────────────
    result_expires=86400,
    result_extended=True,

    # ─── Retry defaults ───────────────────────────────
    task_default_retry_delay=60,
    task_default_max_retries=3,

    # ─── Broker connection ────────────────────────────
    # Silences the CPendingDeprecationWarning that fires on
    # startup and on every reconnect in Celery 5.4+. Will be
    # the default in Celery 6.
    broker_connection_retry_on_startup=True,

    # ─── Queue routing ────────────────────────────────
    task_default_queue="monitoring",
    task_routes={
        "app.tasks.monitor_tasks.*": {"queue": "monitoring"},
        "app.tasks.notification_tasks.*": {"queue": "notifications"},
        # Verification tasks re-use the monitoring queue so the
        # existing worker picks them up. They're LLM-bound, so
        # they naturally serialize behind monitor tasks and don't
        # compete for Ollama's concurrency slots.
        "verification.*": {"queue": "monitoring"},
        # Existing task families that were already routing
        # implicitly. Made explicit so the map is complete.
        "matching.*": {"queue": "monitoring"},
        "embedding.*": {"queue": "monitoring"},
        "recs.*": {"queue": "monitoring"},
    },
)


# ═══════════════════════════════════════════════════════
# Beat schedule
# ═══════════════════════════════════════════════════════
celery_app.conf.beat_schedule = {
    # ─── Monitoring ────────────────────────────────────
    "monitor-megapc": {
        "task": "app.tasks.monitor_tasks.run_site_monitoring",
        "schedule": crontab(minute="*/5"),
        "args": ["megapc"],
        "options": {"expires": 240},
    },
    "monitor-tunisianet": {
        "task": "app.tasks.monitor_tasks.run_site_monitoring",
        "schedule": crontab(minute="*/10"),
        "args": ["tunisianet"],
        "options": {"expires": 540},
    },
    "monitor-sbs": {
        "task": "app.tasks.monitor_tasks.run_site_monitoring",
        "schedule": crontab(minute="*/10"),
        "args": ["sbs"],
        "options": {"expires": 540},
    },
    "monitor-mytek": {
        "task": "app.tasks.monitor_tasks.run_site_monitoring",
        "schedule": crontab(minute="0"),
        "args": ["mytek"],
        "options": {"expires": 3000},
    },
    "monitor-spacenet": {
        "task": "app.tasks.monitor_tasks.run_site_monitoring",
        "schedule": crontab(minute="30"),
        "args": ["spacenet"],
        "options": {"expires": 3000},
    },
    "monitor-skymil": {
        "task": "app.tasks.monitor_tasks.run_site_monitoring",
        "schedule": crontab(minute="*/15"),
        "args": ["skymil"],
        "options": {"expires": 840},
    },
    "cleanup-old-batches": {
        "task": "app.tasks.monitor_tasks.cleanup_old_batches",
        "schedule": crontab(hour=3, minute=0),
        "kwargs": {"days": 30},
    },
    "cleanup-old-alerts": {
        "task": "app.tasks.notification_tasks.cleanup_old_alerts",
        "schedule": crontab(hour=4, minute=0),
        "kwargs": {"days": 90},
    },

    # ─── Matching ─────────────────────────────────────
    "matching-nightly": {
        "task": "matching.nightly",
        "schedule": crontab(hour=3, minute=0),
    },

    # ─── Recommendations ──────────────────────────────
    "recs-refresh-nightly": {
        "task": "recs.refresh_all_users",
        "schedule": crontab(hour=4, minute=0),
    },
    "recs-purge-expired": {
        "task": "recs.purge_expired",
        "schedule": crontab(minute="*/30"),
    },

    # ─── Verification ────────────────────────────────
    # Runs every 6 hours, offset from each other by 30 min so
    # the two LLM-heavy loops don't compete for Ollama slots.
    "verify-canonicals": {
        "task": "verification.verify_canonicals",
        "schedule": crontab(minute=15, hour="*/6"),
        "kwargs": {"limit": 50},
    },
    "verify-recommendations": {
        "task": "verification.verify_recommendations",
        "schedule": crontab(minute=45, hour="*/6"),
        "kwargs": {"limit": 50, "rebuild_on_mismatch": True},
    },
    "sweep-unsent-alerts": {
        "task": "notifications.sweep_unsent_alerts",
        "schedule": crontab(minute="*/15"),
        "kwargs": {"max_age_minutes": 15},
    },
}