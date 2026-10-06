"""Base group ORM — the umbrella table over variant canonicals."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class BaseGroup(Base):
    __tablename__ = "base_groups"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    base_name: Mapped[str] = mapped_column(Text, nullable=False)
    base_brand: Mapped[Optional[str]] = mapped_column(String(200))
    base_category: Mapped[Optional[str]] = mapped_column(String(200))
    base_signature: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )

    variant_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    site_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    listing_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )

    cheapest_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    cheapest_site: Mapped[Optional[str]] = mapped_column(String(50))
    price_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )

    verified_by: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    is_locked: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<BaseGroup {self.base_name[:40]!r} "
            f"variants={self.variant_count}>"
        )