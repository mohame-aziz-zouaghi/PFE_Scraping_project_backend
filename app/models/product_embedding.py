"""SQLAlchemy ORM for product_embeddings (pgvector-backed)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.product import Product

EMBEDDING_DIM = 768


class ProductEmbedding(Base):
    __tablename__ = "product_embeddings"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    product_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=False
    )
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    canonical_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("canonical_products.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    product: Mapped["Product"] = relationship(  # noqa: F821
        "Product", backref="embedding", lazy="noload"
    )

    __table_args__ = (
        UniqueConstraint("product_id", "model", name="uq_product_embedding_model"),
        Index("ix_product_embeddings_product", "product_id"),
    )

    def __repr__(self) -> str:
        return f"<ProductEmbedding product={self.product_id} model={self.model}>"
