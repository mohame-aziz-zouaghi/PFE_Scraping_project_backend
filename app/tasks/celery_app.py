"""
Celery application configuration.

Uses separate Redis DBs for broker and result backend to prevent
cache eviction from wiping task results.

    DB 0 → application cache (progress, sessions, rate limits)
    DB 1 → Celery broker queue
    DB 2 → Celery task results
"""
from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "monitoring",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.monitor_tasks",
        "app.tasks.notification_tasks",
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
    task_time_limit=3600,           # hard kill at 1h
    task_soft_time_limit=3300,      # soft kill at 55min
    worker_prefetch_multiplier=1,   # one task at a time per worker
    worker_max_tasks_per_child=200, # restart worker after 200 tasks
    # ─── Results ──────────────────────────────────────
    result_expires=86400,           # 24h to keep task results queryable
    result_extended=True,           # store args/kwargs with result
    # ─── Retry defaults ───────────────────────────────
    task_default_retry_delay=60,
    task_default_max_retries=3,
    # ─── Queue routing ────────────────────────────────
    task_default_queue="monitoring",
    task_routes={
        "app.tasks.monitor_tasks.*": {"queue": "monitoring"},
        "app.tasks.notification_tasks.*": {"queue": "notifications"},
    },
)

# ═══════════════════════════════════════════════════════
# Beat schedule
# ═══════════════════════════════════════════════════════
# NOTE: Each site gets its own schedule with staggering to prevent overlap.
# Sites running hourly don't collide; frequent sites (skymil) do.
celery_app.conf.beat_schedule = {
    "monitor-megapc": {
        "task": "app.tasks.monitor_tasks.run_site_monitoring",
        "schedule": crontab(minute="*/5"),
        "args": ["megapc"],
        "options": {"expires": 240},   # skip if not picked up in 4 min
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
        "schedule": crontab(minute="0"),         # top of every hour
        "args": ["mytek"],
        "options": {"expires": 3000},
    },
    "monitor-spacenet": {
        "task": "app.tasks.monitor_tasks.run_site_monitoring",
        "schedule": crontab(minute="30"),        # half-hour offset
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
        "schedule": crontab(hour="3", minute="0"),   # daily at 3 AM
        "kwargs": {"days": 30},
    },
    "cleanup-old-alerts": {
        "task": "app.tasks.notification_tasks.cleanup_old_alerts",
        "schedule": crontab(hour="4", minute="0"),   # daily at 4 AM
        "kwargs": {"days": 90},
    },
    "matching-nightly": {
        "task": "matching.nightly",       # NOT full_run
        "schedule": crontab(hour=3, minute=0),
    },
}