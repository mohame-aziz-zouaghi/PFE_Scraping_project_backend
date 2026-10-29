"""
Per-user rate limiters for alert notifications.

Two windows:
    online  — 2 notifications per 1 hour
    offline — 2 emails per 6 hours

Atomic via Lua script: read + increment + set expire in one op.
No race, no "get then set" gaps under load.
"""
from __future__ import annotations

from app.core.logging_config import get_logger

logger = get_logger(__name__)

ONLINE_LIMIT   = 10
ONLINE_WINDOW  = 3600       # 1h

OFFLINE_LIMIT  = 10
OFFLINE_WINDOW = 3 * 3600   # 3h

# Lua: KEYS[1]=key, ARGV[1]=limit, ARGV[2]=window_seconds
# Returns new count. Caller treats count > limit as "exceeded".
_LUA_INCR_WINDOW = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local n = redis.call('INCR', key)
if n == 1 then
    redis.call('EXPIRE', key, window)
end
return n
"""


async def try_consume_online(user_id: str) -> tuple[bool, int]:
    """
    Returns (allowed, current_count).
    allowed=True if within ONLINE_LIMIT for the window.
    """
    return await _try_consume(
        key=f"notif_rate_online:{user_id}",
        limit=ONLINE_LIMIT,
        window=ONLINE_WINDOW,
    )


async def try_consume_offline(user_id: str) -> tuple[bool, int]:
    return await _try_consume(
        key=f"notif_rate_offline:{user_id}",
        limit=OFFLINE_LIMIT,
        window=OFFLINE_WINDOW,
    )


async def _try_consume(key: str, limit: int, window: int) -> tuple[bool, int]:
    from app.db.redis_client import acquire_redis

    try:
        async with acquire_redis() as r:
            count = await r.eval(_LUA_INCR_WINDOW, 1, key, limit, window)
            count = int(count)
            return count <= limit, count
    except Exception as exc:
        logger.warning(
            "notif_rate_limit_redis_failed",
            key=key,
            error=str(exc)[:200],
        )
        # Fail-open: allow the notification. Missing a notif is
        # worse than an occasional extra one.
        return True, 0


async def peek_online(user_id: str) -> int:
    from app.db.redis_client import acquire_redis
    try:
        async with acquire_redis() as r:
            v = await r.get(f"notif_rate_online:{user_id}")
            return int(v) if v else 0
    except Exception:
        return 0


async def peek_offline(user_id: str) -> int:
    from app.db.redis_client import acquire_redis
    try:
        async with acquire_redis() as r:
            v = await r.get(f"notif_rate_offline:{user_id}")
            return int(v) if v else 0
    except Exception:
        return 0