"""
Rate limiting backed by Redis (no external limiter library).

Provides dependency factories:
    rate_limit_anon()   -> anonymous / IP-based
    rate_limit_user()   -> authenticated user
    rate_limit_paid()   -> paid tier

Usage in a router:
    @router.get("/x", dependencies=[Depends(rate_limit_user())])
    async def x(...): ...
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from app.config import settings
from app.core.logging_config import get_logger
from app.db.redis_client import get_redis

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Identifier — per user if authenticated, else per IP
# ═══════════════════════════════════════════════════════
async def user_or_ip_identifier(request: Request) -> str:
    user = getattr(request.state, "user", None)
    if user and getattr(user, "id", None):
        return f"user:{user.id}"

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


# ═══════════════════════════════════════════════════════
# Core limiter — fixed window counter in Redis
# ═══════════════════════════════════════════════════════
async def _check_limit(
    key: str,
    times: int,
    seconds: int,
) -> None:
    """
    Fixed-window rate limit.
    INCR the counter, set TTL on first hit, raise 429 if over.
    """
    redis = await get_redis()
    if redis is None:
        # Fail open — never block traffic if Redis is down
        logger.warning("rate_limit_redis_unavailable", key=key)
        return

    bucket = int(time.time()) // seconds
    redis_key = f"rl:{key}:{bucket}"

    try:
        count = await redis.incr(redis_key)
        if count == 1:
            await redis.expire(redis_key, seconds)
    except Exception as e:
        logger.warning("rate_limit_check_failed", error=str(e), key=key)
        return  # fail open

    if count > times:
        retry_after = seconds - (int(time.time()) % seconds)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


# ═══════════════════════════════════════════════════════
# Dependency factories
# ═══════════════════════════════════════════════════════
def rate_limit_anon(times: Optional[int] = None, seconds: int = 60):
    limit = times or settings.rate_limit_anon_per_minute

    async def _dep(request: Request) -> None:
        ident = await user_or_ip_identifier(request)
        await _check_limit(f"anon:{ident}", limit, seconds)

    return Depends(_dep)


def rate_limit_user(times: Optional[int] = None, seconds: int = 60):
    limit = times or settings.rate_limit_user_per_minute

    async def _dep(request: Request) -> None:
        ident = await user_or_ip_identifier(request)
        await _check_limit(f"user:{ident}", limit, seconds)

    return Depends(_dep)


def rate_limit_paid(times: Optional[int] = None, seconds: int = 60):
    limit = times or settings.rate_limit_paid_per_minute

    async def _dep(request: Request) -> None:
        ident = await user_or_ip_identifier(request)
        await _check_limit(f"paid:{ident}", limit, seconds)

    return Depends(_dep)


def rate_limit(times: int, seconds: int = 60, scope: str = "custom"):
    """Ad-hoc limit for a specific endpoint."""
    async def _dep(request: Request) -> None:
        ident = await user_or_ip_identifier(request)
        await _check_limit(f"{scope}:{ident}", times, seconds)

    return Depends(_dep)