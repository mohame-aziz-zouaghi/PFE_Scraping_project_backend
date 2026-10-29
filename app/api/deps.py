"""
FastAPI dependencies: current user, DB, Redis.
"""
from typing import Optional
from uuid import UUID

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    
)
from app.core.security import decode_token
from app.db.session import get_db
from app.models.user import User

# auto_error=False so we can return 401 ourselves with our own error shape
bearer_scheme = HTTPBearer(auto_error=False)


def _extract_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> Optional[str]:
    """
    Extract JWT token from either:
    1. Authorization: Bearer <token>  header
    2. access_token cookie (for browser sessions)
    """
    if credentials and credentials.credentials:
        return credentials.credentials
    return request.cookies.get("access_token")


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract + validate current user from JWT (header or cookie)."""
    token = _extract_token(request, credentials)
    if not token:
        raise AuthenticationError("Missing authentication")

    # decode_token raises AuthenticationError on any JWT problem
    payload = decode_token(token, expected_type="access")

    user_id_str = payload.get("sub")
    if not user_id_str:
        raise AuthenticationError("Invalid token payload")

    # Guard against malformed UUID strings
    try:
        user_uuid = UUID(user_id_str)
    except (ValueError, TypeError):
        raise AuthenticationError("Invalid user identifier in token")

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()

    # ── Same validation order as before, extended for new flags ──
    if not user:
        raise AuthenticationError("User inactive or not found")
    if user.is_deleted:
        raise AuthenticationError("User inactive or not found")
    if user.is_currently_banned:
        raise AuthenticationError("Account banned")
    if not user.is_active:
        raise AuthenticationError("User inactive or not found")

    # Expose to rate limiter and downstream handlers
    request.state.user = user
    request.state.user_id = user.id
    return user


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Optional auth — returns None if not authenticated.
    Never raises for auth problems (bad token = treated as anonymous).
    """
    token = _extract_token(request, credentials)
    if not token:
        return None
    try:
        return await get_current_user(request, credentials, db)
    except AuthenticationError:
        return None
    except AuthenticationError:
        # Banned user — treat as anonymous so public routes still work
        return None


async def get_current_admin(
    user: User = Depends(get_current_user),
) -> User:
    """Ensure the user is an admin."""
    if user.role not in ("admin", "superadmin") and user.tier != "admin":
        raise AuthorizationError("Admin access required")
    return user



async def require_superadmin(
    user: User = Depends(get_current_user),
) -> User:
    """Gate for the highest-privilege operations.

    Assumes User has either `role == 'superadmin'` or `is_superadmin`.
    Adjust to match your model.
    """
    role = getattr(user, "role", None)
    is_super = getattr(user, "is_superadmin", False)
    if not (is_super or role == "superadmin"):
        raise AuthorizationError("Superadmin privileges required.")
    return user