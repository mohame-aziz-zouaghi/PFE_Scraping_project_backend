# app/models/recommendation.py
"""ORM for the materialized recommendations table."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    product_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
    )

    rank: Mapped[int] = mapped_column(
        SmallInteger, nullable=False
    )
    stage2_score: Mapped[float] = mapped_column(
        Numeric(6, 4), nullable=False
    )
    stage3_score: Mapped[Optional[float]] = mapped_column(
        Numeric(6, 4), nullable=True
    )

    # Now includes: "content" | "brand" | "category" | "price" |
    #               "popularity" | "search" | "trending"
    source: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    reason: Mapped[Optional[str]] = mapped_column(Text)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "product_id",
            name="uq_recommendation_user_product",
        ),
        CheckConstraint(
            "reason IS NULL OR char_length(reason) <= 400",
            name="ck_rec_reason_length",
        ),
        Index(
            "ix_recommendations_user_rank",
            "user_id", "rank",
        ),
        Index("ix_recommendations_expiry", "expires_at"),
        Index(
            "ix_recommendations_user_fresh",
            "user_id", "generated_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Recommendation user={self.user_id} "
            f"product={self.product_id} rank={self.rank}>"
        )