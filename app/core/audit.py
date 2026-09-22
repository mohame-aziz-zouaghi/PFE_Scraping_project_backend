"""Audit logging."""
import json
from typing import Optional
from uuid import UUID

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.models.audit import AuditLog

logger = get_logger(__name__)


def _safe_details(details: Optional[dict]) -> dict:
    """Ensure details is JSON-serializable for JSONB."""
    if not details:
        return {}
    try:
        # Round-trip through JSON to remove unserializable types
        return json.loads(json.dumps(details, default=str))
    except Exception:
        return {"_unserializable": True}


def _get_client_ip(request: Request) -> Optional[str]:
    """Extract client IP, respecting X-Forwarded-For if behind a proxy."""
    # Behind proxy (Cloudflare / nginx)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()[:45]  # fit INET column
    # Fallback
    if request.client:
        return request.client.host[:45]
    return None


async def log_action(
    db: AsyncSession,
    request: Optional[Request] = None,
    action: str = "",
    resource: Optional[str] = None,
    user_id: Optional[UUID] = None,
    details: Optional[dict] = None,
):
    ip = None
    ua = None
    if request is not None:
        forwarded = request.headers.get("x-forwarded-for")
        ip = forwarded.split(",")[0].strip() if forwarded else (
            request.client.host if request.client else None
        )
        ua = request.headers.get("user-agent")

    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource=resource,
        ip=ip,
        user_agent=ua,
        details=details or {},
    )
    db.add(entry)
    await db.flush()
    return entry