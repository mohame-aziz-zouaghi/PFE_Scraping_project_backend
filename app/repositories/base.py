"""
Small, reusable async SQLAlchemy repository primitives.

Subclasses set `model` (and optionally `id_field`) and get basic
CRUD without repeating boilerplate.
"""
from typing import Generic, List, Optional, TypeVar

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    # Override in subclasses if the primary key isn't named "id"
    id_field: str = "id"

    def __init__(self, db: AsyncSession, model: type[ModelT]):
        self.db = db
        self.model = model

    # ─── Read ─────────────────────────────────────────
    async def get(self, identifier) -> Optional[ModelT]:
        """Fetch one row by primary key."""
        col = getattr(self.model, self.id_field)
        result = await self.db.execute(
            select(self.model).where(col == identifier)
        )
        return result.scalar_one_or_none()

    async def count(self) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(self.model)
        )
        return int(result.scalar_one() or 0)

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[ModelT]:
        result = await self.db.execute(
            select(self.model).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    # ─── Write ────────────────────────────────────────
    async def add(self, entity: ModelT) -> ModelT:
        """
        Add + flush. Caller decides when to commit.
        Returns the same instance (with server-side defaults applied).
        """
        self.db.add(entity)
        await self.db.flush()
        return entity

    async def update(self, entity: ModelT, **fields) -> ModelT:
        """Update fields on a managed entity and flush."""
        for k, v in fields.items():
            setattr(entity, k, v)
        await self.db.flush()
        return entity

    async def delete(self, entity: ModelT) -> None:
        await self.db.delete(entity)
        await self.db.flush()

    async def delete_by_id(self, identifier) -> bool:
        """Delete by primary key. Returns True if a row was deleted."""
        col = getattr(self.model, self.id_field)
        result = await self.db.execute(
            sa_delete(self.model).where(col == identifier)
        )
        await self.db.flush()
        return bool(result.rowcount)