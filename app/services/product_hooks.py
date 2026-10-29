# app/services/product_hooks.py
"""
SQLAlchemy event hooks on Product.

Only UPDATEs are handled here — INSERTs are handled by the
PostgreSQL trigger (`notify_product_inserted`), which fires on
every INSERT regardless of whether the writer used the ORM or
raw SQL. Duplicating the INSERT path here would enqueue every
new product twice.

On UPDATE, we enqueue only if a field that affects the
embedding text or canonicalization actually changed. Updates
to `last_checked`, `price_valid_until`, `last_price_change`,
etc. don't trigger re-embedding.

Import this module once at app startup to activate the hooks.
Importing is idempotent.
"""
from __future__ import annotations

from sqlalchemy import event, inspect as sa_inspect

from app.core.logging_config import get_logger
from app.models.product import Product

logger = get_logger(__name__)

_REGISTERED = False

# Fields whose change requires re-canonicalizing or re-embedding.
_TRIGGER_FIELDS = {
    "name",
    "brand",
    "category",
    "short_description",
    "canonical_id",     # if the linker rewrites it, re-embed
    "is_enabled",       # disabled → skip; re-enabled → embed again
}


def register_product_hooks() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    @event.listens_for(Product, "after_update")
    def _after_update(mapper, connection, target):  # noqa: ARG001
        """
        Fires after an ORM UPDATE flush. Inspects which attrs
        actually changed and enqueues jobs only if one of the
        embedding-relevant fields moved.
        """
        state = sa_inspect(target)
        changed = {
            attr.key for attr in state.attrs
            if attr.history.has_changes()
        }
        if not (changed & _TRIGGER_FIELDS):
            return

        _enqueue_after_update(target.id)

    _REGISTERED = True
    logger.info("rag_product_hooks_registered")


def _enqueue_after_update(product_id) -> None:
    """
    Re-embed on update. Canonical re-linking happens too if
    name/brand/category changed — the linker is idempotent and
    fast, so enqueueing both unconditionally is simpler than
    distinguishing the update kind.
    """
    from app.services.embedding_queue import embedding_queue
    embedding_queue.enqueue("canonical_link", product_id)
    embedding_queue.enqueue("embed_product", product_id)