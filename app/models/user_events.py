# app/models/user_events.py
"""ORM for user_events — append-only signal log."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


# Event types the API accepts. Anything else is rejected before
# it hits the DB. The SQL CHECK constraint is the last line of
# defence.
EVENT_TYPES = (
    "view",
    "click",
    "dwell",
    "wishlist",
    "add_to_cart",
    "search",
)

# Default weights used for scoring. Higher = stronger signal.
# These are the *base* weights; the tier multiplier (fresh /
# today / recent / old) is applied on top of them at ranking
# time so a fresh wishlist outranks an old add_to_cart.
EVENT_WEIGHTS: dict[str, float] = {
    "view": 1.0,
    "click": 2.0,
    "dwell": 1.0,
    "wishlist": 5.0,
    "search": 5.0,
    "add_to_cart": 8.0,
}


class UserEvent(Base):
    __tablename__ = "user_events"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )

    product_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="SET NULL"),
        nullable=True,
    )
    canonical_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("canonical_products.id", ondelete="SET NULL"),
        nullable=True,
    )
    base_group_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("base_groups.id", ondelete="SET NULL"),
        nullable=True,
    )

    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict,
        server_default="{}",
    )
    weight: Mapped[float] = mapped_column(
        Numeric(5, 2), nullable=False, default=1.0,
        server_default="1.0",
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('view','click','dwell',"
            "'wishlist','add_to_cart','search')",
            name="ck_user_events_type",
        ),
        Index(
            "ix_user_events_user_time",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_user_events_product_time",
            "product_id",
            "created_at",
        ),
        Index(
            "ix_user_events_type_time",
            "event_type",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<UserEvent {self.event_type} user={self.user_id} "
            f"product={self.product_id}>"
        )