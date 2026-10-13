"""Auth endpoints."""
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import settings
from app.core.audit import log_action
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.logging_config import get_logger
from app.core.rate_limit import rate_limit_anon, rate_limit_user
from app.core.security import (
    ACCESS_COOKIE,
    REFRESH_COOKIE,
    clear_auth_cookies,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    set_auth_cookies,
    verify_password,
)
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    MessageResponse,
    RegisterResponse,
    ResendVerificationRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    VerifyEmailRequest,
)
from app.services.auth_service import AuthService

logger = get_logger(__name__)
router = APIRouter()


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


# ═══════════════════════════════════════════════════════
# POST /register
# ═══════════════════════════════════════════════════════
@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[rate_limit_anon(times=30, seconds=3600)],
)
async def register(
    request: Request,
    payload: UserRegister,
    db: AsyncSession = Depends(get_db),
) -> RegisterResponse:
    """Register a new user and send a verification email.

    Response includes `verification_expires_at` so the frontend can
    run a live countdown on the register wizard's final step.
    """
    svc = AuthService(db)

    response, raw_token = await svc.register(
        email=payload.email,
        username=payload.username,
        password=payload.password,
        display_name=payload.display_name,
        phone=payload.phone,
        ip=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        source="web",
    )

    await log_action(
        db,
        request,
        "user.register",
        f"user:{response.user_id}",
        response.user_id,
    )
    await db.commit()

    # Send AFTER commit — SMTP failure won't roll back the user
    user = await db.get(User, response.user_id)
    if user is not None:
        sent = await svc.send_verification_email(user, raw_token)
        logger.info(
            "user_registered",
            user_id=str(user.id),
            email_sent=sent,
        )

    return response


# ═══════════════════════════════════════════════════════
# POST /login
# ═══════════════════════════════════════════════════════
@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[rate_limit_anon(times=30, seconds=60)],
)
async def login(
    request: Request,
    response: Response,
    payload: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    svc = AuthService(db)
    user, access_token, refresh_token = await svc.login(
        email=payload.email,
        password=payload.password,
        ip=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    set_auth_cookies(response, access_token, refresh_token)
    return TokenResponse(
        access_token=access_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


# ═══════════════════════════════════════════════════════
# POST /refresh
# ═══════════════════════════════════════════════════════
@router.post(
    "/refresh",
    response_model=TokenResponse,
    dependencies=[rate_limit_anon(times=30, seconds=60)],
)
async def refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise AuthenticationError("Missing refresh token")

    payload = decode_token(token, expected_type="refresh")
    try:
        user_uuid = UUID(payload.get("sub"))
    except (ValueError, TypeError):
        clear_auth_cookies(response)
        raise AuthenticationError("Invalid refresh token")

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()

    if not user or not user.can_login:
        clear_auth_cookies(response)
        raise AuthenticationError("Invalid refresh token")

    access_token = create_access_token(
        str(user.id),
        extra_claims={"tier": user.tier, "role": user.role},
    )
    new_refresh = create_refresh_token(str(user.id))
    set_auth_cookies(response, access_token, new_refresh)

    user.last_seen_at = datetime.now(timezone.utc)
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


# ═══════════════════════════════════════════════════════
# POST /logout
# ═══════════════════════════════════════════════════════
@router.post("/logout")
async def logout(
    response: Response,
    user: User = Depends(get_current_user),
):
    clear_auth_cookies(response)
    logger.info("user_logged_out", user_id=str(user.id))
    return {"message": "Logged out"}


# ═══════════════════════════════════════════════════════
# GET /me
# ═══════════════════════════════════════════════════════
@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)) -> UserResponse:
    """Return the current user's public profile.

    The response_model filters out everything not listed on
    UserResponse — sensitive fields (tokens, hashes, IPs, ban
    metadata) never leave the server.
    """
    return UserResponse.model_validate(user)


# ═══════════════════════════════════════════════════════
# POST /verify-email
# ═══════════════════════════════════════════════════════
@router.post(
    "/verify-email",
    response_model=MessageResponse,
    dependencies=[rate_limit_anon(times=20, seconds=60)],
)
async def verify_email(
    request: Request,
    payload: VerifyEmailRequest,
    db: AsyncSession = Depends(get_db),
):
    svc = AuthService(db)
    user = await svc.verify_email(payload.token)
    await log_action(
        db, request, "auth.email_verified", f"user:{user.id}", user.id
    )
    await db.commit()
    return MessageResponse(message="Email verified successfully")


# ═══════════════════════════════════════════════════════
# GET /verification-status  (used by the register wizard)
# ═══════════════════════════════════════════════════════
@router.get(
    "/verification-status",
    dependencies=[rate_limit_anon(times=60, seconds=60)],
)
async def verification_status(
    email: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return whether an email has been verified.

    Only exposes a boolean — does not leak whether the account
    exists in a distinguishable way. If the email is unknown,
    returns `is_verified=False` (same as an unverified account).
    """
    result = await db.execute(
        select(User.is_verified).where(User.email == email.lower())
    )
    is_verified = result.scalar_one_or_none()
    return {"is_verified": bool(is_verified)}


# ═══════════════════════════════════════════════════════
# POST /resend-verification
# ═══════════════════════════════════════════════════════
@router.post(
    "/resend-verification",
    response_model=MessageResponse,
    dependencies=[rate_limit_anon(times=3, seconds=3600)],
)
async def resend_verification(
    request: Request,
    payload: ResendVerificationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Always returns 200 — does not reveal whether the email is
    registered.
    """
    svc = AuthService(db)
    await svc.resend_verification(payload.email)
    return MessageResponse(
        message=(
            "If that email exists and isn't verified, "
            "a new link was sent."
        )
    )


# ═══════════════════════════════════════════════════════
# POST /forgot-password
# ═══════════════════════════════════════════════════════
@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    dependencies=[rate_limit_anon(times=20, seconds=3600)],
)
async def forgot_password(
    request: Request,
    payload: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    svc = AuthService(db)
    await svc.request_password_reset(payload.email)
    return MessageResponse(
        message=(
            "If that email is registered, a reset link has been sent."
        )
    )


# ═══════════════════════════════════════════════════════
# POST /reset-password
# ═══════════════════════════════════════════════════════
@router.post(
    "/reset-password",
    response_model=MessageResponse,
    dependencies=[rate_limit_anon(times=10, seconds=3600)],
)
async def reset_password(
    request: Request,
    response: Response,
    payload: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    svc = AuthService(db)
    user = await svc.reset_password(payload.token, payload.new_password)
    await log_action(
        db, request, "auth.password_reset", f"user:{user.id}", user.id
    )
    await db.commit()
    clear_auth_cookies(response)
    return MessageResponse(
        message="Password reset successfully. Please log in."
    )


# ═══════════════════════════════════════════════════════
# POST /change-password (authenticated)
# ═══════════════════════════════════════════════════════
@router.post(
    "/change-password",
    response_model=MessageResponse,
    dependencies=[rate_limit_user(times=5, seconds=3600)],
)
async def change_password(
    request: Request,
    response: Response,
    payload: ChangePasswordRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    svc = AuthService(db)
    await svc.change_password(
        user, payload.current_password, payload.new_password
    )
    await log_action(
        db, request, "auth.password_changed", f"user:{user.id}", user.id
    )
    await db.commit()
    clear_auth_cookies(response)
    return MessageResponse(
        message="Password changed. Please log in again."
    )