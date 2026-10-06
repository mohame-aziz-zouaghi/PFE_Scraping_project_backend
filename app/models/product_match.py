"""ORM models for cross-site product matching."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

EMBEDDING_DIM = 768


class ProductMatchPair(Base):
    """Audit trail of pairwise matching decisions."""
    __tablename__ = "product_match_pairs"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    product_a_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
    )
    product_b_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
    )
    canonical_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("canonical_products.id", ondelete="CASCADE"),
    )
    cosine_distance: Mapped[float] = mapped_column(
        Numeric(6, 5), nullable=False
    )
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    decision_source: Mapped[str] = mapped_column(String(20), nullable=False)
    variant_match: Mapped[Optional[bool]] = mapped_column(Boolean)
    token_overlap: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    llm_reason: Mapped[Optional[str]] = mapped_column(Text)
    llm_confidence: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("product_a_id", "product_b_id",
                         name="uq_match_pair"),
        Index("ix_match_pairs_canonical", "canonical_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<MatchPair {self.product_a_id} ↔ {self.product_b_id} "
            f"{self.decision}>"
        )