"""
Async Redis client.

Simple lazy-init singleton. No dynamic pooling, no reaper, no
lease watchdog — just the plain redis.asyncio client that works.

- Lazy initialization (safe for concurrent async access)
- Optional password support
- Health check
- Graceful shutdown
- Backward-compatible: every public symbol the previous versions
  exported is still here (get_redis, close_redis, check_redis_health,
  get_redis_pool_stats, cache_get/set/delete, acquire_redis,
  with_redis).
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from typing import Any, AsyncGenerator, Optional, TypeVar

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════
_client: Optional[redis.Redis] = None
_pool: Optional[ConnectionPool] = None
_lock: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    """
    Lazily create the lock so it binds to the running loop, not to
    whatever loop happened to exist at import time (Windows --reload
    footgun).
    """
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


# ═══════════════════════════════════════════════════════
# Client lifecycle
# ═══════════════════════════════════════════════════════
async def get_redis() -> redis.Redis:
    """
    Return the singleton Redis client, initializing on first call.
    Safe for concurrent async access.

    Never raises on connection failure — returns a client whose
    first command will fail. /ready is what reports Redis health.
    """
    global _client, _pool

    if _client is not None:
        return _client

    async with _get_lock():
        # Double-check after acquiring the lock
        if _client is not None:
            return _client

        _pool = ConnectionPool.from_url(
            settings.effective_redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.redis_max_connections,
            socket_keepalive=True,
            socket_connect_timeout=settings.redis_connect_timeout,
            socket_timeout=settings.redis_socket_timeout,
            health_check_interval=30,
            retry_on_timeout=True,
            retry_on_error=[ConnectionError, TimeoutError],
        )
        _client = redis.Redis(connection_pool=_pool)

        # Verify connection once. Failure is logged, not raised —
        # the app must boot even if Redis is down (degraded mode).
        try:
            await _client.ping()
            logger.info("redis_connected")
        except Exception as e:
            logger.error("redis_ping_failed | %s", e)

        return _client


async def close_redis() -> None:
    """Close the Redis client and connection pool. Idempotent."""
    global _client, _pool

    if _client is not None:
        try:
            await _client.aclose()
        except Exception as e:
            logger.warning("redis_close_failed | %s", e)
        finally:
            _client = None

    if _pool is not None:
        try:
            await _pool.disconnect()
        except Exception as e:
            logger.warning("redis_pool_disconnect_failed | %s", e)
        finally:
            _pool = None

    logger.info("redis_disconnected")


# ═══════════════════════════════════════════════════════
# Health + stats
# ═══════════════════════════════════════════════════════
async def check_redis_health(client=None) -> bool:
    try:
        r = client or await get_redis()
        if r is None:
            return False
        # Short timeout so /ready can't hang
        await asyncio.wait_for(r.ping(), timeout=2.0)
        return True
    except asyncio.TimeoutError:
        logger.warning("redis_health_check_timeout")
        return False
    except Exception as e:
        logger.warning("redis_health_check_failed | %s", e)
        return False


def get_redis_pool_stats() -> dict:
    """Return Redis connection pool stats (informational)."""
    if _pool is None:
        return {"initialized": False}
    return {
        "initialized": True,
        "max_connections": getattr(_pool, "max_connections", None),
        "created_connections": (
            len(getattr(_pool, "_available_connections", []))
            + len(getattr(_pool, "_in_use_connections", []))
        ),
        "available_connections": len(
            getattr(_pool, "_available_connections", [])
        ),
        "in_use_connections": len(
            getattr(_pool, "_in_use_connections", [])
        ),
    }


# ═══════════════════════════════════════════════════════
# Lease — thin pass-through
# ═══════════════════════════════════════════════════════
# Callers that already use `async with acquire_redis() as r:` keep
# working. The lease wraps the SHARED client — no per-lease
# connection is held. Commands borrow a connection from the pool
# per-command and return it immediately (redis-py's normal model).

class RedisLease:
    """Pass-through wrapper around the shared client."""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if self._closed:
                raise RuntimeError("lease already closed")
            return await attr(*args, **kwargs)

        return _wrapped

    # Sync pass-throughs — `__getattr__` would wrap these as async,
    # which breaks `pipeline()` and `pubsub()` construction.
    def pipeline(self, *a, **k):
        if self._closed:
            raise RuntimeError("lease already closed")
        return self._client.pipeline(*a, **k)

    def pubsub(self, *a, **k):
        if self._closed:
            raise RuntimeError("lease already closed")
        return self._client.pubsub(*a, **k)

    async def aclose(self) -> None:
        self._closed = True


@contextlib.asynccontextmanager
async def acquire_redis(
    *,
    db: int = 0,
    op_timeout: Optional[float] = None,
) -> AsyncGenerator[RedisLease, None]:
    """
    Return a lease over the shared client.

    Signature is compatible with the previous pooling version, but
    there is no per-lease connection and no watchdog — commands fail
    fast via the pool's own `socket_timeout`. `db` and `op_timeout`
    are accepted but ignored (only DB 0 is used by this client).

    Rationale: the shared `redis.asyncio.Redis` client already
    borrows and returns a connection per command. There is no
    benefit to a per-lease connection for this workload.
    """
    client = await get_redis()
    lease = RedisLease(client)
    try:
        yield lease
    finally:
        await lease.aclose()


def with_redis(db: int = 0, op_timeout: Optional[float] = None):
    """Decorator: injects `r=` (a RedisLease) as a keyword."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*a, **k):
            async with acquire_redis(db=db, op_timeout=op_timeout) as r:
                return await fn(*a, r=r, **k)
        return wrapper
    return deco


# ═══════════════════════════════════════════════════════
# Convenience helpers
# ═══════════════════════════════════════════════════════
async def cache_get(key: str) -> Optional[str]:
    client = await get_redis()
    if client is None:
        return None
    return await client.get(key)


async def cache_set(key: str, value: str, ttl: int = 300) -> bool:
    client = await get_redis()
    if client is None:
        return False
    await client.setex(key, ttl, value)
    return True


async def cache_delete(key: str) -> bool:
    client = await get_redis()
    if client is None:
        return False
    return bool(await client.delete(key))