# app/models/canonical_embedding.py
"""
ORM for canonical_embeddings — pgvector index over canonical_products.

One row per canonical product. `source_text` is built at nightly
refresh time from the canonical row:

    "<canonical_name> — available at <site_count> sites, cheapest at
     <cheapest_site> for <cheapest_price> DT"
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.types import Vector


class CanonicalEmbedding(Base):
    __tablename__ = "canonical_embeddings"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
    )

    canonical_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("canonical_products.id", ondelete="CASCADE"),
        nullable=False,
    )

    model: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )
    source_text: Mapped[str] = mapped_column(
        Text, nullable=False,
    )

    source_hash: Mapped[str] = mapped_column(
    String(64), nullable=False,
)

    # Same note as HelpEmbedding: the vector column is declared in
    # the migration, not here. Reads/writes go through raw SQL.
    embedding: Mapped[list[float]] = mapped_column(
        Vector(768), nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_canonical_embeddings_canonical", "canonical_id",
        ),
        Index("ix_canonical_embeddings_model", "model"),
    )

    def __repr__(self) -> str:
        return (
            f"<CanonicalEmbedding canonical={self.canonical_id} "
            f"model={self.model}>"
        )