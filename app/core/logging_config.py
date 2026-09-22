"""
Structured logging configuration.

Routing:
- Terminal        → everything (human-readable, styled)
- logs/errors.log → ERROR + CRITICAL only (styled text)
- logs/api.log    → API/auth/monitoring logs at INFO+ (styled text)
- logs/app.log    → every event (styled text) — full audit trail

Rotation:
- Each file rotates at 10 MB, keeps 5 backups.

File format:
    YYYY-MM-DD HH:MM:SS.mmm | LEVEL | logger
      key          : value
      key          : value
    -------------------------------------------------------------------

Usage:
    from app.core.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("user_login", user_id=..., request_id=...)
"""
import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Iterable, Optional

import structlog

from app.config import settings


# ═══════════════════════════════════════════════════════
# Log directory
# ═══════════════════════════════════════════════════════
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════
# Loggers whose INFO-level events go to logs/api.log
# ═══════════════════════════════════════════════════════
API_LOGGER_PREFIXES = (
    "app.api",                 # route handlers
    "app.core.audit",          # audit trail
    "app.core.honeypot",       # bot detection
    "app.core.security",       # JWT / password
    "app.core.rate_limit",     # rate limiter
    "app.services",            # business logic
    "app.tasks",               # celery task wrappers
    "app.middleware",          # request middleware
)


# ═══════════════════════════════════════════════════════
# Filters
# ═══════════════════════════════════════════════════════
class MaxLevelFilter(logging.Filter):
    """Allow only records at or below `max_level`."""
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


class MinLevelFilter(logging.Filter):
    """Allow only records at or above `min_level`."""
    def __init__(self, min_level: int):
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


class ApiLoggerFilter(logging.Filter):
    """
    Allow records if:
    - level >= WARNING (always keep warnings/errors), OR
    - logger name starts with one of API_LOGGER_PREFIXES (keep INFO+ API events)
    """
    def __init__(self, prefixes: Iterable[str]):
        super().__init__()
        self.prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return any(record.name.startswith(p) for p in self.prefixes)


# ═══════════════════════════════════════════════════════
# Shared processors
# ═══════════════════════════════════════════════════════
def add_app_context(logger, method_name, event_dict):
    event_dict["environment"] = settings.environment
    event_dict["app_version"] = settings.app_version
    return event_dict


def _base_processors() -> list:
    """Processors shared by every output."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def _build_formatter(renderer) -> structlog.stdlib.ProcessorFormatter:
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_base_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


# ═══════════════════════════════════════════════════════
# NEW: Styled file renderer — date/time header + aligned body
# ═══════════════════════════════════════════════════════
class FormattedFileRenderer:
    """
    Renders log entries for files as:

        2026-09-16 10:53:22.172 | INFO  | app.api.v1.auth
          event        : user_logged_in
          user_id      : abc-123-def
          request_id   : req-xyz-789
        -------------------------------------------------------------------
    """

    LINE_WIDTH = 68
    KEY_WIDTH = 14

    # Keys that appear first (in this order), the rest alphabetical
    PRIORITY_KEYS = ("event", "message", "error", "error_type", "path", "method")

    LEVEL_LABELS = {
        "debug": "DEBUG",
        "info": "INFO ",
        "warning": "WARN ",
        "error": "ERROR",
        "critical": "CRIT ",
    }

    def __call__(self, logger, method_name, event_dict):
        # Pop reserved keys
        timestamp = event_dict.pop("timestamp", "")
        level = event_dict.pop("level", "info").lower()
        logger_name = event_dict.pop("logger", "")
        event = event_dict.pop("event", "")
        message = event_dict.pop("message", None)

        # ISO → human-readable
        ts_display = timestamp.replace("T", " ").replace("Z", "")
        if "." in ts_display:
            head, rest = ts_display.split(".", 1)
            ts_display = f"{head}.{rest[:3]}"

        level_label = self.LEVEL_LABELS.get(level, level.upper())
        header = f"{ts_display} | {level_label} | {logger_name}"

        # ─── Order payload ───────────────────────────
        rows = []
        seen = set()

        if event:
            rows.append(("event", event))
            seen.add("event")

        if message is not None:
            rows.append(("message", message))
            seen.add("message")

        for k in self.PRIORITY_KEYS:
            if k in event_dict and k not in seen:
                rows.append((k, event_dict.pop(k)))
                seen.add(k)

        for k in sorted(event_dict.keys()):
            if k not in seen:
                rows.append((k, event_dict[k]))

        # ─── Render body ─────────────────────────────
        lines = [header]
        for key, value in rows:
            rendered = self._render_value(value)
            if "\n" in rendered:
                first, *rest = rendered.split("\n")
                lines.append(f"  {key:<{self.KEY_WIDTH}}: {first}")
                for r in rest:
                    lines.append(" " * (2 + self.KEY_WIDTH + 2) + r)
            else:
                lines.append(f"  {key:<{self.KEY_WIDTH}}: {rendered}")

        lines.append("-" * self.LINE_WIDTH)
        return "\n".join(lines) + "\n"

    def _render_value(self, value) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            if "\n" in value:
                return value.replace("\n", "\\n")
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)


# ═══════════════════════════════════════════════════════
# File handler factory
# ═══════════════════════════════════════════════════════
def _make_file_handler(
    filename: str,
    level: int,
    filters: Optional[list[logging.Filter]] = None,
) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,          # don't create file until first log
    )
    handler.setLevel(level)
    # Files use the styled renderer
    handler.setFormatter(
        _build_formatter(FormattedFileRenderer())
    )
    for f in filters or []:
        handler.addFilter(f)
    return handler


# ═══════════════════════════════════════════════════════
# Main configuration
# ═══════════════════════════════════════════════════════
def configure_logging() -> None:
    """Configure structlog + stdlib logging."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # ─── Renderer for console (UNCHANGED) ─────────────
    if settings.log_format == "json":
        console_renderer = structlog.processors.JSONRenderer()
    else:
        console_renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )

    # ─── structlog global config ──────────────────────
    structlog.configure(
        processors=_base_processors() + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ─── Formatters ───────────────────────────────────
    console_formatter = _build_formatter(console_renderer)

    # ─── Handlers ─────────────────────────────────────
    # 1. Console: everything at INFO+ (styled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # 2. errors.log: ERROR and CRITICAL only (styled)
    errors_handler = _make_file_handler(
        filename="errors.log",
        level=logging.ERROR,
    )

    # 3. api.log: API events (INFO+ from API loggers), plus all WARNING+
    api_handler = _make_file_handler(
        filename="api.log",
        level=logging.INFO,
        filters=[ApiLoggerFilter(API_LOGGER_PREFIXES)],
    )

    # 4. app.log: everything at INFO+ (full audit trail)
    app_handler = _make_file_handler(
        filename="app.log",
        level=log_level,
    )

    # ─── Root logger ──────────────────────────────────
    root = logging.getLogger()
    root.handlers.clear()
    for h in (console_handler, errors_handler, api_handler, app_handler):
        root.addHandler(h)
    root.setLevel(log_level)

    # ─── Third-party logger routing ───────────────────
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(log_level)

    access_lg = logging.getLogger("uvicorn.access")
    access_lg.handlers.clear()
    access_lg.propagate = True
    access_lg.setLevel(logging.INFO)

    # SQLAlchemy noise: only emit at WARNING+
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)

    # Asyncpg / redis: quiet unless warnings
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)

    # WatchFiles (uvicorn reloader): silence
    logging.getLogger("watchfiles").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════
# Logger factory
# ═══════════════════════════════════════════════════════
def get_logger(name: str = __name__):
    """Get a structlog logger that plays nice with stdlib."""
    return structlog.get_logger(name)