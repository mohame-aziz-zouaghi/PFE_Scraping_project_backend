# app/models/user_profile.py
"""ORM for user_profiles — aggregated taste."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )

    total_events: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    distinct_products: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    # Each entry:
    #   {"name": str, "weight": float, "tier": "fresh"|"today"|"recent"|"long"|"mixed"}
    preferred_categories: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    preferred_brands: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )

    price_min: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    price_max: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    price_mean: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    price_p25: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    price_p75: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))

    last_event_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    is_cold_start: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )

    # Recent search queries (last N distinct terms, freshest first).
    recent_searches: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_user_profiles_updated", "updated_at"),
        Index("ix_user_profiles_expires_at", "expires_at"),
        Index(
            "ix_user_profiles_rebuild_scan",
            "expires_at", "updated_at",
            postgresql_where=(is_cold_start.is_(False)),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<UserProfile user={self.user_id} "
            f"events={self.total_events} cold={self.is_cold_start}>"
        )