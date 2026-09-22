"""JWT + password hashing + cookie session management."""
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Any, Optional
from uuid import UUID

from fastapi import Response
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings
from app.core.exceptions import AuthenticationError

# ─── Password hashing ───
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password cannot be empty")
    if len(password.encode("utf-8")) > 72:
        raise ValueError("Password too long (max 72 bytes)")
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed:
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


# ─── JWT ───
def create_access_token(
    subject: str,
    extra_claims: Optional[dict] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta
        or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def create_refresh_token(
    subject: str, expires_delta: Optional[timedelta] = None
) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta
        or timedelta(days=settings.refresh_token_expire_days)
    )
    payload = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "refresh",
    }
    return jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def decode_token(token: str, expected_type: str = "access") -> dict:
    if not token:
        raise AuthenticationError("Missing token")
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {e}")
    if payload.get("type") != expected_type:
        raise AuthenticationError(
            f"Invalid token type: expected {expected_type}"
        )
    return payload


# ═══════════════════════════════════════════════════════
# COOKIE SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════

# Cookie names — constants so we never typo them
ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"

# Refresh cookie scoped to refresh endpoint only (minimizes attack surface)
REFRESH_COOKIE_PATH = "/api/v1/auth/refresh"


def _cookie_kwargs(path: str, max_age: int, strict: bool = False) -> dict:
    """
    Build common cookie kwargs.

    - httponly: JS can't read it (XSS protection)
    - secure: HTTPS-only in production
    - samesite: 'strict' for refresh, 'lax' for access
    - path: restricts cookie to specific routes
    - max_age: matches JWT expiry
    """
    return {
        "httponly": True,
        "secure": settings.environment == "production",
        "samesite": "strict" if strict else "lax",
        "path": path,
        "max_age": max_age,
    }


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
) -> None:
    """
    Attach access + refresh cookies to the response.

    Called from /login and /refresh.
    """
    # Access cookie — scoped to entire site
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=access_token,
        **_cookie_kwargs(
            path="/",
            max_age=settings.access_token_expire_minutes * 60,
            strict=False,
        ),
    )

    # Refresh cookie — scoped to refresh endpoint only
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=refresh_token,
        **_cookie_kwargs(
            path=REFRESH_COOKIE_PATH,
            max_age=settings.refresh_token_expire_days * 24 * 3600,
            strict=True,
        ),
    )


def clear_auth_cookies(response: Response) -> None:
    """
    Delete both auth cookies.

    Called from /logout and on authentication failures.
    """
    response.delete_cookie(
        key=ACCESS_COOKIE,
        path="/",
    )
    response.delete_cookie(
        key=REFRESH_COOKIE,
        path=REFRESH_COOKIE_PATH,
    )


    # ═══════════════════════════════════════════════════════
# Opaque token helpers (email verify, password reset)
# ═══════════════════════════════════════════════════════
def generate_opaque_token(nbytes: int = 32) -> str:
    """URL-safe random token, ~43 chars for 32 bytes."""
    return secrets.token_urlsafe(nbytes)


def hash_opaque_token(token: str) -> str:
    """SHA-256 hex — what we store in DB. Raw token goes only in the email."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    import hmac
    return hmac.compare_digest(a, b)