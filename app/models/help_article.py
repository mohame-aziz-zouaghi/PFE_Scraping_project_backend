# app/models/help_article.py
"""
ORM for help_articles — curated FAQ content for the RAG assistant.

Two flavours live in this table, distinguished by `category`:

  • category = 'general'       → anonymous-suggestion questions
                                 (pre-cached, no account context needed).
  • category = 'account'       → logged-in-only questions
                                 (never cached for anonymous users).
  • category = 'pricing'       → price / discount / history questions.
  • category = 'availability'  → stock / on-order / restock questions.
  • category = 'recommendation'→ how recommendations work.
  • category = 'other'         → fallback bucket.

The anon suggestion chips in the frontend only render rows where
`locale = <user_locale>` AND `category = 'general'` AND
`is_published = true`.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


# Categories the API accepts for help articles. Anything else is
# rejected before the row hits the DB.
HELP_CATEGORIES = (
    "general",
    "account",
    "pricing",
    "availability",
    "recommendation",
    "other",
)

# Locales the API accepts. Mirrors the CHECK constraint.
HELP_LOCALES = ("en", "fr", "ar")


class HelpArticle(Base):
    __tablename__ = "help_articles"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    slug: Mapped[str] = mapped_column(
        String(120), nullable=False,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(
        String(80), nullable=False,
    )
    locale: Mapped[str] = mapped_column(
        String(8), nullable=False, server_default="en",
    )
    is_published: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true",
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
        UniqueConstraint(
            "slug", "locale",
            name="uq_help_articles_slug_locale",
        ),
        CheckConstraint(
            "category IN ('general','account','pricing',"
            "'availability','recommendation','other')",
            name="ck_help_articles_category",
        ),
        CheckConstraint(
            "locale IN ('en','fr','ar')",
            name="ck_help_articles_locale",
        ),
        Index(
            "ix_help_articles_category",
            "category",
        ),
        Index(
            "ix_help_articles_published",
            "locale", "is_published",
            postgresql_where=(is_published.is_(True)),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<HelpArticle {self.slug} locale={self.locale} "
            f"cat={self.category}>"
        )