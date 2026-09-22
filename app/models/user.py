"""User model — full auth / ban / activity schema."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import INET, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.base import Base, TimestampMixin


# Boolean columns and their None-coercion default
_TRUE_DEFAULT_BOOLS = {"is_active", "notify_email"}
_FALSE_DEFAULT_BOOLS = {"is_verified", "is_banned", "is_deleted", "notify_push"}


class User(Base, TimestampMixin):
    __tablename__ = "users"

    # ─── Identity ────────────────────────────────────────
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    username: Mapped[str] = mapped_column(
        String(100), unique=True, index=True, nullable=False
    )
    display_name: Mapped[Optional[str]] = mapped_column(String(150))
    avatar_url: Mapped[Optional[str]] = mapped_column(Text)
    phone: Mapped[Optional[str]] = mapped_column(String(30))
    locale: Mapped[str] = mapped_column(
        String(10), default="en", server_default="en", nullable=False
    )
    timezone: Mapped[str] = mapped_column(
        String(50),
        default="Africa/Tunis",
        server_default="Africa/Tunis",
        nullable=False,
    )

    # ─── Auth ────────────────────────────────────────────
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    password_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    password_reset_token: Mapped[Optional[str]] = mapped_column(String(255))
    password_reset_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )

    # ─── Role / tier ─────────────────────────────────────
    role: Mapped[str] = mapped_column(
        String(30), default="user", server_default="user", nullable=False
    )
    tier: Mapped[str] = mapped_column(
        String(20), default="free", server_default="free", nullable=False
    )
    tier_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )

    # ─── Verification ────────────────────────────────────
    is_verified: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    email_verify_token: Mapped[Optional[str]] = mapped_column(String(255))
    email_verify_sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    phone_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )

    # ─── Status flags ────────────────────────────────────
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )

    # ─── Login / lockout ─────────────────────────────────
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    last_login_ip: Mapped[Optional[str]] = mapped_column(INET)
    last_login_user_agent: Mapped[Optional[str]] = mapped_column(Text)
    last_failed_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    failed_login_attempts: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    locked_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    lockout_reason: Mapped[Optional[str]] = mapped_column(String(100))

    # ─── Banning ─────────────────────────────────────────
    is_banned: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    banned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    banned_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    ban_reason: Mapped[Optional[str]] = mapped_column(String(100))
    ban_notes: Mapped[Optional[str]] = mapped_column(Text)
    banned_by: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
    )

    # ─── Soft delete ─────────────────────────────────────
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    deletion_reason: Mapped[Optional[str]] = mapped_column(String(100))

    # ─── Activity / metrics ──────────────────────────────
    login_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    api_request_count: Mapped[int] = mapped_column(
        BigInteger, default=0, server_default="0", nullable=False
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    signup_ip: Mapped[Optional[str]] = mapped_column(INET)
    signup_user_agent: Mapped[Optional[str]] = mapped_column(Text)
    signup_source: Mapped[Optional[str]] = mapped_column(String(50))

    # ─── Preferences ─────────────────────────────────────
    notify_email: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )
    notify_push: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    alert_threshold_pct: Mapped[float] = mapped_column(
        Numeric(6, 2), default=5.0, server_default="5.0", nullable=False
    )
    theme: Mapped[str] = mapped_column(
        String(20), default="system", server_default="system", nullable=False
    )

    # ─── Extras ──────────────────────────────────────────
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        server_default="'{}'::jsonb",
        nullable=False,
    )

    # ─── Self-referential relationship ───────────────────
    banned_by_user: Mapped[Optional["User"]] = relationship(
        "User", remote_side=[id], foreign_keys=[banned_by], lazy="noload"
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('user','moderator','admin','superadmin')",
            name="ck_users_role",
        ),
        CheckConstraint(
            "tier IN ('free','paid','admin')", name="ck_users_tier"
        ),
        CheckConstraint(
            "ban_reason IS NULL OR ban_reason IN "
            "('spam','abuse','fraud','tos_violation','manual','security')",
            name="ck_users_ban_reason",
        ),
        Index("ix_users_role", "role"),
        Index("ix_users_tier", "tier"),
        Index(
            "ix_users_is_banned", "is_banned",
            postgresql_where="is_banned = TRUE",
        ),
        Index(
            "ix_users_is_deleted", "is_deleted",
            postgresql_where="is_deleted = TRUE",
        ),
    )

    # ─── Convenience helpers ─────────────────────────────
    @property
    def is_admin(self) -> bool:
        return self.role in ("admin", "superadmin")

    @property
    def is_staff(self) -> bool:
        return self.role in ("moderator", "admin", "superadmin")

    @property
    def can_login(self) -> bool:
        now = datetime.now(timezone.utc)
        if not self.is_active or self.is_deleted or self.is_banned:
            return False
        if self.locked_until and self.locked_until > now:
            return False
        if self.banned_until and self.banned_until > now:
            return False
        return True

    @property
    def is_currently_banned(self) -> bool:
        if not self.is_banned:
            return False
        if self.banned_until is None:
            return True
        return self.banned_until > datetime.now(timezone.utc)

    @property
    def is_currently_locked(self) -> bool:
        return bool(
            self.locked_until
            and self.locked_until > datetime.now(timezone.utc)
        )

    # ─── Validators — coerce None to safe defaults ───────
    @validates(*_TRUE_DEFAULT_BOOLS, *_FALSE_DEFAULT_BOOLS)
    def _coerce_bool_defaults(self, key: str, value):
        if value is None:
            return key in _TRUE_DEFAULT_BOOLS
        return bool(value)

    @validates("role", "tier")
    def _coerce_role_tier(self, key: str, value):
        if value is None:
            return {"role": "user", "tier": "free"}[key]
        return value

    @validates("locale", "timezone", "theme")
    def _coerce_str_defaults(self, key: str, value):
        if value is None:
            return {
                "locale": "en",
                "timezone": "Africa/Tunis",
                "theme": "system",
            }[key]
        return value

    @validates(
        "failed_login_attempts", "login_count", "api_request_count",
    )
    def _coerce_int_defaults(self, key: str, value):
        if value is None:
            return 0
        return int(value)

    @validates("alert_threshold_pct")
    def _coerce_numeric_defaults(self, key: str, value):
        if value is None:
            return 5.0
        return float(value)

    @validates("metadata_")
    def _coerce_metadata(self, key: str, value):
        if value is None:
            return {}
        return value

    def __repr__(self) -> str:
        return (
            f"<User {self.username} ({self.email}) "
            f"role={self.role} tier={self.tier}>"
        )