"""Honeypot endpoints — trap bots and ban their IPs."""
import ipaddress
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Request
from redis.asyncio import Redis

from app.config import settings
from app.core.exceptions import HoneypotTriggered
from app.core.logging_config import get_logger

logger = get_logger(__name__)

BAN_DURATION_HOURS = 24
BAN_KEY_PREFIX = "banned_ip:"

# Trusted networks — never ban these (avoid self-lockout)
_TRUSTED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # IPv4 localhost
    ipaddress.ip_network("::1/128"),          # IPv6 localhost
    ipaddress.ip_network("10.0.0.0/8"),       # RFC1918
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
]


def _is_trusted(ip: str) -> bool:
    """Return True if the IP is in a trusted network."""
    try:
        addr = ipaddress.ip_address(ip)
        return any(addr in net for net in _TRUSTED_NETWORKS)
    except ValueError:
        return False


def _get_client_ip(request: Request) -> Optional[str]:
    """Extract the true client IP behind proxies."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


async def block_ip(redis: Redis, ip: str, reason: str = "honeypot") -> None:
    """Add IP to ban list with TTL. Skip trusted networks."""
    if _is_trusted(ip):
        logger.info("ip_ban_skipped_trusted", ip=ip)
        return
    key = f"{BAN_KEY_PREFIX}{ip}"
    try:
        await redis.setex(key, timedelta(hours=BAN_DURATION_HOURS), reason)
        logger.warning("ip_banned", ip=ip, reason=reason)
    except Exception as e:
        logger.error("ip_ban_failed", ip=ip, error=str(e))


async def is_ip_banned(redis: Redis, ip: str) -> bool:
    """Return True if the IP is currently banned."""
    try:
        return await redis.exists(f"{BAN_KEY_PREFIX}{ip}") > 0
    except Exception:
        # Fail open — never block on Redis failure
        return False


async def unban_ip(redis: Redis, ip: str) -> bool:
    """Manually remove a ban (useful for admins)."""
    try:
        removed = await redis.delete(f"{BAN_KEY_PREFIX}{ip}")
        return removed > 0
    except Exception:
        return False


def register_honeypot_routes(router: APIRouter) -> None:
    """Register fake endpoints only bots will find."""
    if not settings.honeypot_enabled:
        return

    @router.get("/internal/admin/users", include_in_schema=False)
    @router.post("/internal/admin/users", include_in_schema=False)
    @router.get("/.env", include_in_schema=False)
    @router.get("/backup.sql", include_in_schema=False)
    @router.get("/wp-admin", include_in_schema=False)
    @router.get("/.git/config", include_in_schema=False)
    async def honeypot(request: Request):
        redis: Optional[Redis] = getattr(request.app.state, "redis", None)
        ip = _get_client_ip(request)
        path = request.url.path
        user_agent = request.headers.get("user-agent", "")[:200]

        if redis is not None and ip:
            await block_ip(redis, ip, reason=f"honeypot:{path}")

        logger.warning(
            "honeypot_triggered",
            ip=ip,
            path=path,
            user_agent=user_agent,
        )
        raise HoneypotTriggered("Not found")