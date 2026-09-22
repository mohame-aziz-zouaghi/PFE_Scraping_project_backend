"""
Redis-backed JSON cache for LLM responses and embeddings.

Every entry is a key + JSON value + TTL. Keys are namespaced so different
features never collide. Fails open — a Redis outage returns None, never raises.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from app.config import settings
from app.core.logging_config import get_logger
from app.db.redis_client import get_redis

logger = get_logger(__name__)


def make_key(namespace: str, *parts: Any) -> str:
    """
    Build a deterministic cache key.

    Example:
        make_key("llm", "recommend", user_id, product_id)
        → "cache:llm:7f3a...b2c1"
    """
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"cache:{namespace}:{digest}"


class Cache:
    """Async JSON cache backed by Redis. Fails open (returns None on error)."""

    @staticmethod
    async def get(key: str) -> Optional[Any]:
        if not settings.ai_cache_enabled:
            return None
        try:
            redis = await get_redis()
            if redis is None:
                return None
            raw = await redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("cache_get_failed", key=key, error=str(e))
            return None

    @staticmethod
    async def set(key: str, value: Any, ttl_seconds: int) -> bool:
        if not settings.ai_cache_enabled:
            return False
        try:
            redis = await get_redis()
            if redis is None:
                return False
            await redis.set(key, json.dumps(value, default=str), ex=ttl_seconds)
            return True
        except Exception as e:
            logger.warning("cache_set_failed", key=key, error=str(e))
            return False

    @staticmethod
    async def delete(key: str) -> None:
        try:
            redis = await get_redis()
            if redis is not None:
                await redis.delete(key)
        except Exception as e:
            logger.warning("cache_delete_failed", key=key, error=str(e))

    @staticmethod
    async def get_or_set(key: str, ttl_seconds: int, producer):
        """
        Return cached value, or call `producer()` and cache the result.

        Usage:
            result = await Cache.get_or_set(key, 3600, lambda: expensive_call())
        """
        cached = await Cache.get(key)
        if cached is not None:
            logger.debug("cache_hit", key=key)
            return cached

        value = await producer()
        if value is not None:
            await Cache.set(key, value, ttl_seconds)
        return value
