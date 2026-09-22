"""User repository."""
from datetime import datetime
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, db: AsyncSession):
        super().__init__(db, User)

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.email == email.strip().lower())
        )
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.username == username.strip())
        )
        return result.scalar_one_or_none()

    async def get_by_identity(
        self, email: str, username: str
    ) -> Optional[User]:
        """Find a user by email OR username (for uniqueness checks)."""
        result = await self.db.execute(
            select(User).where(
                or_(
                    User.email == email.strip().lower(),
                    User.username == username.strip(),
                )
            )
        )
        return result.scalar_one_or_none()

    async def update_last_login(
        self, user: User, when: Optional[datetime] = None
    ) -> User:
        user.last_login_at = when or datetime.now(tz=None)
        user.failed_login_attempts = 0
        user.locked_until = None
        await self.db.flush()
        return user

    async def increment_failed_login(
        self, user: User, lockout_until: Optional[datetime] = None
    ) -> User:
        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        if lockout_until:
            user.locked_until = lockout_until
        await self.db.flush()
        return user

    async def list_active(
        self, limit: int = 100, offset: int = 0
    ) -> list[User]:
        result = await self.db.execute(
            select(User)
            .where(User.is_active.is_(True))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())