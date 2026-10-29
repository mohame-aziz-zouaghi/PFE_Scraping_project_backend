# app/services/help_article_hooks.py
"""
SQLAlchemy event hooks that keep help_embeddings in sync with
help_articles.

Registers three hooks:

    after_insert  → enqueue embedding job
    after_update  → enqueue embedding job
    after_delete  → no-op (help_embeddings cascades on delete)

Import this module once at app startup to activate the hooks.
Importing is idempotent — the hooks are only registered the
first time.
"""
from __future__ import annotations

from sqlalchemy import event

from app.core.logging_config import get_logger
from app.models.help_article import HelpArticle

logger = get_logger(__name__)

_REGISTERED = False


def register_help_article_hooks() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    @event.listens_for(HelpArticle, "after_insert")
    def _after_insert(mapper, connection, target):  # noqa: ARG001
        _enqueue(target.id)

    @event.listens_for(HelpArticle, "after_update")
    def _after_update(mapper, connection, target):  # noqa: ARG001
        _enqueue(target.id)

    _REGISTERED = True
    logger.info("rag_help_article_hooks_registered")


def _enqueue(article_id) -> None:
    # Lazy import to avoid loading the queue at module import
    # time (which would fail before the event loop exists).
    from app.services.embedding_queue import embedding_queue
    embedding_queue.enqueue("help", article_id)