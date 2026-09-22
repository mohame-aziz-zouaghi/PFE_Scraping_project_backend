"""Auth service — register, login, refresh, ban, lockout."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.audit import log_action
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.logging_config import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    generate_opaque_token,
    hash_opaque_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.services.email_service import EmailService

logger = get_logger(__name__)


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ═══════════════════════════════════════════════════════
    # Registration
    # ═══════════════════════════════════════════════════════
    async def register(
        self,
        *,
        email: str,
        username: str,
        password: str,
        ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        source: str = "web",
    ) -> User:
        existing = await self.db.execute(
            select(User).where(
                or_(
                    User.email == email.lower(),
                    User.username == username,
                )
            )
        )
        if existing.scalar_one_or_none():
            raise ConflictError("Email or username already registered")

        user = User(
            email=email.lower(),
            username=username,
            display_name=username,
            hashed_password=hash_password(password),
            role="user",
            tier="free",
            is_active=True,
            is_verified=False,
            signup_ip=ip,
            signup_user_agent=user_agent,
            signup_source=source,
            password_changed_at=datetime.now(timezone.utc),
        )
        self.db.add(user)
        await self.db.flush()

        logger.info(
            "user_registered",
            user_id=str(user.id),
            email=user.email,
            ip=ip,
        )
        return user

    # ═══════════════════════════════════════════════════════
    # Login — every failure case handled
    # ═══════════════════════════════════════════════════════
    async def login(
        self,
        *,
        email: str,
        password: str,
        ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> tuple[User, str, str]:
        """Returns (user, access_token, refresh_token)."""
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        user = result.scalar_one_or_none()
        now = datetime.now(timezone.utc)

        # ── Not found — generic error, no enumeration
        if not user:
            await log_action(
                self.db,
                action="auth.login_failed",
                resource=None,
                details={"email": email, "reason": "not_found", "ip": ip},
            )
            raise AuthenticationError("Invalid credentials")

        # ── Soft deleted
        if user.is_deleted:
            await log_action(
                self.db, action="auth.login_failed",
                resource=f"user:{user.id}",
                details={"reason": "deleted"}, user_id=user.id,
            )
            raise AuthenticationError("Account is no longer available")

        # ── Banned (permanent or temporary)
        if user.is_currently_banned:
            await log_action(
                self.db, action="auth.login_failed",
                resource=f"user:{user.id}",
                details={"reason": "banned", "ban_reason": user.ban_reason},
                user_id=user.id,
            )
            raise AuthenticationError(
                f"Account is banned: {user.ban_reason or 'contact support'}"
            )

        # ── Banned but ban expired — auto-clear
        if user.is_banned and not user.is_currently_banned:
            user.is_banned = False
            user.banned_at = None
            user.banned_until = None
            user.ban_reason = None
            user.ban_notes = None

        # ── Currently locked
        if user.is_currently_locked:
            await log_action(
                self.db, action="auth.login_failed",
                resource=f"user:{user.id}",
                details={
                    "reason": "locked",
                    "locked_until": user.locked_until.isoformat(),
                },
                user_id=user.id,
            )
            raise AuthenticationError("Account locked. Try again later.")

        # ── Lock expired — clear it
        if user.locked_until and not user.is_currently_locked:
            user.locked_until = None
            user.lockout_reason = None
            user.failed_login_attempts = 0

        # ── Inactive
        if not user.is_active:
            await log_action(
                self.db, action="auth.login_failed",
                resource=f"user:{user.id}",
                details={"reason": "inactive"}, user_id=user.id,
            )
            raise AuthenticationError("Account inactive")

        # ── Password check
                # ── Password check
        if not verify_password(password, user.hashed_password):
            user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
            user.last_failed_login_at = now

            if user.failed_login_attempts >= settings.max_failed_login_attempts:
                user.locked_until = now + timedelta(
                    minutes=settings.account_lockout_minutes
                )
                user.lockout_reason = "too_many_failed_attempts"
                logger.warning(
                    "account_locked",
                    user_id=str(user.id),
                    attempts=user.failed_login_attempts,
                    until=user.locked_until.isoformat(),
                )

            await self.db.commit()
            await log_action(
                self.db, action="auth.login_failed",
                resource=f"user:{user.id}",
                details={
                    "reason": "bad_password",
                    "attempts": user.failed_login_attempts,
                },
                user_id=user.id,
            )
            raise AuthenticationError("Invalid credentials")

        # ─────────────────────────────────────────────────────
        # ✨ NEW — Email verification gate
        # Only reached after a correct password, so this message
        # does not leak whether the account exists.
        # ─────────────────────────────────────────────────────
        if not user.is_verified:
            await log_action(
                self.db, action="auth.login_blocked_unverified",
                resource=f"user:{user.id}", user_id=user.id,
                details={"email": user.email, "ip": ip},
            )
            await self.db.commit()
            raise AuthenticationError(
                "Email not verified. Check your inbox or request a new link."
            )

        # ── Success — reset counters, record metadata
        user.failed_login_attempts = 0
        user.locked_until = None
        user.lockout_reason = None
        user.last_login_at = now
        user.last_login_ip = ip
        user.last_login_user_agent = user_agent
        user.last_seen_at = now
        user.login_count = (user.login_count or 0) + 1

        access_token = create_access_token(
            str(user.id),
            extra_claims={"tier": user.tier, "role": user.role},
        )
        refresh_token = create_refresh_token(str(user.id))

        await log_action(
            self.db, action="auth.login_success",
            resource=f"user:{user.id}", user_id=user.id,
            details={"ip": ip},
        )
        await self.db.commit()

        logger.info(
            "user_logged_in",
            user_id=str(user.id),
            tier=user.tier,
            role=user.role,
        )
        return user, access_token, refresh_token

    # ═══════════════════════════════════════════════════════
    # Refresh — validates user is still allowed
    # ═══════════════════════════════════════════════════════
    async def refresh(self, user_id: UUID) -> tuple[User, str, str]:
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user or not user.can_login:
            raise AuthenticationError("Session no longer valid")

        access_token = create_access_token(
            str(user.id),
            extra_claims={"tier": user.tier, "role": user.role},
        )
        refresh_token = create_refresh_token(str(user.id))

        user.last_seen_at = datetime.now(timezone.utc)
        await self.db.commit()
        return user, access_token, refresh_token

    # ═══════════════════════════════════════════════════════
    # Ban / Unban (admin actions)
    # ═══════════════════════════════════════════════════════
    async def ban_user(
        self,
        *,
        target: User,
        admin: User,
        reason: str,
        notes: Optional[str] = None,
        duration_hours: Optional[int] = None,
    ) -> User:
        now = datetime.now(timezone.utc)
        target.is_banned = True
        target.banned_at = now
        target.banned_until = (
            now + timedelta(hours=duration_hours)
            if duration_hours else None
        )
        target.ban_reason = reason
        target.ban_notes = notes
        target.banned_by = admin.id

        await log_action(
            self.db,
            action="user.banned",
            resource=f"user:{target.id}",
            user_id=admin.id,
            details={
                "reason": reason,
                "notes": notes,
                "duration_hours": duration_hours,
            },
        )
        await self.db.commit()
        logger.warning(
            "user_banned",
            target=str(target.id),
            by=str(admin.id),
            reason=reason,
        )
        return target

    async def unban_user(self, *, target: User, admin: User) -> User:
        target.is_banned = False
        target.banned_at = None
        target.banned_until = None
        target.ban_reason = None
        target.ban_notes = None
        target.banned_by = None

        await log_action(
            self.db, action="user.unbanned",
            resource=f"user:{target.id}", user_id=admin.id,
        )
        await self.db.commit()
        logger.info("user_unbanned", target=str(target.id), by=str(admin.id))
        return target

    # ═══════════════════════════════════════════════════════
    # Soft delete / restore
    # ═══════════════════════════════════════════════════════
    async def soft_delete(self, *, target: User, reason: str = "user_request"):
        target.is_deleted = True
        target.deleted_at = datetime.now(timezone.utc)
        target.deletion_reason = reason
        target.is_active = False
        await self.db.commit()
        logger.info("user_soft_deleted", target=str(target.id), reason=reason)
        return target

    async def restore(self, *, target: User):
        target.is_deleted = False
        target.deleted_at = None
        target.deletion_reason = None
        target.is_active = True
        await self.db.commit()
        return target



        # ═══════════════════════════════════════════════════════
    # Email verification
    # ═══════════════════════════════════════════════════════
    async def issue_verification_token(self, user: User) -> str:
        """
        Generate a fresh verification token, persist its hash,
        return the RAW token (caller emails it).
        """
        raw = generate_opaque_token()
        user.email_verify_token = hash_opaque_token(raw)
        user.email_verify_sent_at = datetime.now(timezone.utc)
        await self.db.flush()
        return raw

    async def send_verification_email(self, user: User, raw_token: str) -> bool:
        """Send the verification email using an already-issued raw token."""
        return await EmailService.send_verification(user, raw_token)

    async def verify_email(self, raw_token: str) -> User:
        hashed = hash_opaque_token(raw_token)
        result = await self.db.execute(
            select(User).where(User.email_verify_token == hashed)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise AuthenticationError("Invalid or expired verification link")
        if user.is_deleted or user.is_currently_banned:
            raise AuthenticationError("Invalid or expired verification link")
        if user.is_verified:
            # Idempotent — already verified
            user.email_verify_token = None
            await self.db.commit()
            return user

        now = datetime.now(timezone.utc)
        sent = user.email_verify_sent_at
        if sent and (now - sent) > timedelta(
            hours=settings.email_verify_token_ttl_hours
        ):
            user.email_verify_token = None
            await self.db.commit()
            raise AuthenticationError("Verification link expired")

        user.is_verified = True
        user.email_verified_at = now
        user.email_verify_token = None
        user.email_verify_sent_at = None

        await log_action(
            self.db, action="user.email_verified",
            resource=f"user:{user.id}", user_id=user.id,
        )
        await self.db.commit()
        logger.info("email_verified", user_id=str(user.id))
        return user

    async def resend_verification(self, email: str) -> None:
        """
        Always returns success — never leak whether an email exists.
        """
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        user = result.scalar_one_or_none()

        # Silent no-op in all these cases
        if not user or user.is_deleted or user.is_currently_banned:
            return
        if user.is_verified:
            return

        # Throttle: don't resend if issued in last 60 seconds
        if user.email_verify_sent_at:
            delta = datetime.now(timezone.utc) - user.email_verify_sent_at
            if delta.total_seconds() < 60:
                return

        await self.send_verification_email(user)

    # ═══════════════════════════════════════════════════════
    # Password reset
    # ═══════════════════════════════════════════════════════
    async def request_password_reset(self, email: str) -> None:
        """
        Always returns success — never leak whether an email exists.
        """
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        user = result.scalar_one_or_none()

        if not user or user.is_deleted or user.is_currently_banned:
            return
        if not user.is_active:
            return

        # Throttle: don't reissue if issued in last 60 seconds
        now = datetime.now(timezone.utc)
        if user.password_reset_expires_at:
            issued_at = user.password_reset_expires_at - timedelta(
                minutes=settings.password_reset_token_ttl_minutes
            )
            if (now - issued_at).total_seconds() < 60:
                return

        raw = generate_opaque_token()
        user.password_reset_token = hash_opaque_token(raw)
        user.password_reset_expires_at = now + timedelta(
            minutes=settings.password_reset_token_ttl_minutes
        )
        await self.db.flush()

        await log_action(
            self.db, action="auth.password_reset_requested",
            resource=f"user:{user.id}", user_id=user.id,
        )
        await self.db.commit()

        await EmailService.send_password_reset(user, raw)

    async def reset_password(self, raw_token: str, new_password: str) -> User:
        hashed = hash_opaque_token(raw_token)
        result = await self.db.execute(
            select(User).where(User.password_reset_token == hashed)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise AuthenticationError("Invalid or expired reset link")
        if user.is_deleted or user.is_currently_banned or not user.is_active:
            raise AuthenticationError("Invalid or expired reset link")

        now = datetime.now(timezone.utc)
        if not user.password_reset_expires_at or user.password_reset_expires_at < now:
            user.password_reset_token = None
            user.password_reset_expires_at = None
            await self.db.commit()
            raise AuthenticationError("Reset link expired")

        # Rotate password
        user.hashed_password = hash_password(new_password)
        user.password_changed_at = now

        # Consume the token
        user.password_reset_token = None
        user.password_reset_expires_at = None

        # Security: unlock the account and reset counters on successful reset
        user.failed_login_attempts = 0
        user.locked_until = None
        user.lockout_reason = None

        await log_action(
            self.db, action="auth.password_reset_success",
            resource=f"user:{user.id}", user_id=user.id,
        )
        await self.db.commit()
        logger.info("password_reset", user_id=str(user.id))
        return user

    # ═══════════════════════════════════════════════════════
    # Change password (logged-in user)
    # ═══════════════════════════════════════════════════════
    async def change_password(
        self, user: User, current_password: str, new_password: str
    ) -> User:
        if not verify_password(current_password, user.hashed_password):
            raise AuthenticationError("Current password is incorrect")

        if current_password == new_password:
            raise ConflictError("New password must differ from current")

        now = datetime.now(timezone.utc)
        user.hashed_password = hash_password(new_password)
        user.password_changed_at = now

        # Invalidate any pending reset
        user.password_reset_token = None
        user.password_reset_expires_at = None

        await log_action(
            self.db, action="auth.password_changed",
            resource=f"user:{user.id}", user_id=user.id,
        )
        await self.db.commit()
        logger.info("password_changed", user_id=str(user.id))
        return user