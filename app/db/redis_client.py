"""
Async Redis client.

- Lazy initialization
- Health check
- Graceful shutdown
- Works with or without password
- Connection pool diagnostics
"""
import asyncio
import logging
from typing import Optional

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)


# Module-level singleton
_client: Optional[redis.Redis] = None
_pool: Optional[ConnectionPool] = None
_lock = asyncio.Lock()  # guards double-initialization


async def get_redis() -> redis.Redis:
    """
    Return the singleton Redis client, initializing on first call.
    Safe for concurrent async access.
    """
    global _client, _pool
    if _client is not None:
        return _client

    async with _lock:
        # Double-check after acquiring lock
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
            health_check_interval=30,       # auto-ping idle connections
            retry_on_timeout=True,
            retry_on_error=[ConnectionError, TimeoutError],
        )
        _client = redis.Redis(connection_pool=_pool)

        # Verify connection
        try:
            await _client.ping()
            logger.info("redis_connected | url_scheme=redis")
        except Exception as e:
            logger.error("redis_ping_failed | %s", e)
            # Don't raise — /ready will report failure.
            # This lets the app boot even if Redis is down.

        return _client


async def close_redis() -> None:
    """Close the Redis client and connection pool."""
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


async def check_redis_health(client=None) -> bool:
    try:
        redis = client or await get_redis()
        if redis is None:
            return False
        await redis.ping()
        return True
    except Exception as e:
        logger.warning(f"redis_health_check_failed | {e}")
        return False


def get_redis_pool_stats() -> dict:
    """Return Redis connection pool stats."""
    if _pool is None:
        return {"initialized": False}
    return {
        "initialized": True,
        "max_connections": _pool.max_connections,
        "created_connections": _pool._created_connections
        if hasattr(_pool, "_created_connections")
        else None,
        "available_connections": len(_pool._available_connections)
        if hasattr(_pool, "_available_connections")
        else None,
        "in_use_connections": len(_pool._in_use_connections)
        if hasattr(_pool, "_in_use_connections")
        else None,
    }


# ─────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────
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