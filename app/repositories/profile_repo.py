# app/repositories/profile_repo.py
"""Read/write for user_profiles."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.recommendation_config import PROFILE
from app.models.user_profile import UserProfile


class ProfileRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, user_id: UUID) -> Optional[UserProfile]:
        """Unconditional read. Use `get_fresh` on the hot path."""
        return await self.db.get(UserProfile, user_id)

    async def get_fresh(
        self, user_id: UUID
    ) -> Optional[UserProfile]:
        """Return the profile only if it hasn't expired yet."""
        now = datetime.now(timezone.utc)
        stmt = select(UserProfile).where(
            UserProfile.user_id == user_id,
            UserProfile.expires_at > now,
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def upsert(
        self, user_id: UUID, **fields
    ) -> UserProfile:
        """
        Upsert a profile row.

        If `expires_at` isn't supplied, stamp it with the default
        TTL so callers don't have to remember to set it.
        """
        fields.setdefault(
            "expires_at",
            datetime.now(timezone.utc)
            + timedelta(hours=PROFILE.TTL_HOURS),
        )

        existing = await self.get(user_id)
        if existing is None:
            profile = UserProfile(user_id=user_id, **fields)
            self.db.add(profile)
            await self.db.flush()
            return profile

        for key, value in fields.items():
            setattr(existing, key, value)
        await self.db.flush()
        return existing

    async def delete(self, user_id: UUID) -> None:
        await self.db.execute(
            delete(UserProfile).where(
                UserProfile.user_id == user_id
            )
        )

    async def list_expired_user_ids(
        self, *, limit: int = 5000
    ) -> list[UUID]:
        """
        Users whose profile is expired, most recently active first.
        Used by the nightly refresh task.
        """
        now = datetime.now(timezone.utc)
        stmt = (
            select(UserProfile.user_id)
            .where(
                UserProfile.is_cold_start.is_(False),
                UserProfile.expires_at <= now,
            )
            .order_by(UserProfile.updated_at.desc())
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).all()
        return [r[0] for r in rows]