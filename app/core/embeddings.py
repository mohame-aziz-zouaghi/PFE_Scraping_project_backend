# app/core/embeddings.py
"""
Thin shim so RAG code can `from app.core.embeddings import embed_text`.

The real implementation lives in
`app.services.embedding_service.EmbeddingService`. This module
exists because a lot of existing code imports from
`app.core.embeddings` — we don't want to change every caller, and
we don't want two implementations of the same function drifting
apart.

Previous state: this module contained a stub with
`SEMANTIC_ENABLED = False` that always returned None. That stub
is now removed. The real service is always called.

To keep the debug visibility (the RAG pipeline treats a None
return as "embedding unavailable"), set EMBEDDING_DEBUG=1 in the
environment to raise on failure instead.
"""
from __future__ import annotations

import os
from typing import Optional
from uuid import UUID

from app.core.logging_config import get_logger
from app.services.embedding_service import (
    EmbeddingError,
    EmbeddingService,
)

logger = get_logger(__name__)

_DEBUG = os.environ.get("EMBEDDING_DEBUG", "0") == "1"


async def embed_text(
    text: str,
    *,
    db=None,
    user_id: Optional[UUID] = None,
    endpoint: str = "rag",
) -> Optional[list[float]]:
    """
    Embed a single string via the real embedding service.

    Returns None on failure — which is what the RAG pipeline
    expects (a None return triggers the "assistant unavailable"
    refusal). Set EMBEDDING_DEBUG=1 to raise instead, so the
    real exception surfaces.
    """
    if not text or not text.strip():
        return None

    try:
        svc = EmbeddingService(
            db=db, user_id=user_id, endpoint=endpoint
        )
        return await svc.embed_one(text)
    except EmbeddingError as exc:
        logger.warning(
            "embed_text_failed", error=str(exc)[:200]
        )
        if _DEBUG:
            raise
        return None
    except Exception as exc:
        logger.exception(
            "embed_text_unhandled", error=str(exc)[:200]
        )
        if _DEBUG:
            raise
        return None


async def embed_batch(
    texts: list[str],
    *,
    db=None,
    user_id: Optional[UUID] = None,
    endpoint: str = "rag",
) -> list[list[float]]:
    """
    Embed a batch. Matches the old stub's signature so nothing
    that imported `embed_batch` breaks.

    Returns a list of vectors (one per input). Skips any input
    that fails — the caller can detect this by length.
    """
    if not texts:
        return []
    try:
        svc = EmbeddingService(
            db=db, user_id=user_id, endpoint=endpoint
        )
        return await svc.embed_batch(texts)
    except EmbeddingError as exc:
        logger.warning(
            "embed_batch_failed", error=str(exc)[:200]
        )
        if _DEBUG:
            raise
        return []
    except Exception as exc:
        logger.exception(
            "embed_batch_unhandled", error=str(exc)[:200]
        )
        if _DEBUG:
            raise
        return []