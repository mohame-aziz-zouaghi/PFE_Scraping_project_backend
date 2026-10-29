# app/models/help_embedding.py
"""
ORM for help_embeddings — pgvector index over help_articles.

One row per article. `source_text` = title + "\n\n" + body, embedded
with the same model the rest of the app uses (`settings.active_embedding_model`).
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


class  HelpEmbedding(Base):
    __tablename__ = "help_embeddings"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
    )

    article_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("help_articles.id", ondelete="CASCADE"),
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

    # The vector column is declared in the migration as
    # `vector(768)`. SQLAlchemy has no native type for it, so we
    # bind it through a raw SQL type. The column is written and
    # read only via raw SQL in `rag_retrieval.py`; the ORM
    # definition exists so the model is importable and
    # discoverable, not so it can be queried column-by-column.
    embedding: Mapped[list[float]] = mapped_column(
        Vector(768), nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_help_embeddings_article", "article_id"),
        Index("ix_help_embeddings_model", "model"),
        # The HNSW index itself is created in the migration; it's
        # not declared here because SQLAlchemy doesn't know about
        # the `hnsw` access method.
    )

    def __repr__(self) -> str:
        return (
            f"<HelpEmbedding article={self.article_id} "
            f"model={self.model}>"
        )