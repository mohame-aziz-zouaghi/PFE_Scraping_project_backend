"""
Activity heartbeat middleware.

Updates users.last_seen_at on authenticated requests. Throttled:
writes at most once per HEARTBEAT_INTERVAL_SECONDS per user via a
Redis SETNX to skip the write if we already updated recently.

Kept as a separate middleware (not in auth deps) so every
authenticated request updates it — including ones that don't
touch the DB for other reasons.
"""
from __future__ import annotations

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging_config import get_logger

logger = get_logger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 60
_HEARTBEAT_KEY = "user_heartbeat"


class ActivityMiddleware(BaseHTTPMiddleware):
    """
    After the response is sent, if the request was authenticated,
    update the user's last_seen_at at most once per minute.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        response = await call_next(request)

        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            return response

        # Only heartbeat on successful, real API calls
        if request.url.path.startswith("/api/v1/") and response.status_code < 400:
            try:
                await _heartbeat(str(user_id))
            except Exception as exc:
                logger.debug(
                    "activity_heartbeat_failed",
                    error=str(exc)[:200],
                )
        return response


async def _heartbeat(user_id: str) -> None:
    """
    Update last_seen_at at most once per HEARTBEAT_INTERVAL_SECONDS.
    """
    from app.db.redis_client import acquire_redis
    from app.db import acquire_db
    from sqlalchemy import text

    # Skip the DB write if we've already updated recently.
    try:
        async with acquire_redis() as r:
            key = f"{_HEARTBEAT_KEY}:{user_id}"
            # SET NX with TTL — returns True if this is the first
            # heartbeat inside the interval.
            first = await r.set(
                key, "1",
                nx=True,
                ex=HEARTBEAT_INTERVAL_SECONDS,
            )
            if not first:
                return
    except Exception:
        # If Redis is down, fall through and just write the DB —
        # one write per request is acceptable during an outage.
        pass

    try:
        async with acquire_db() as db:
            await db.execute(
                text(
                    "UPDATE users SET last_seen_at = now() "
                    "WHERE id = :uid"
                ),
                {"uid": user_id},
            )
    except Exception as exc:
        logger.debug(
            "activity_db_update_failed",
            error=str(exc)[:200],
        )