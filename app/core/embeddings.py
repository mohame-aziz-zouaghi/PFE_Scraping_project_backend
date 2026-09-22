"""
Optional embedding provider.

Semantic search only activates when:
1. This module defines a working `embed_text()`
2. `SEMANTIC_ENABLED` is True
3. `products.embedding` is populated

Until then, search falls back to text search.
"""
from typing import Optional

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# Set to True once you have a real embedding provider wired up
SEMANTIC_ENABLED = False

# Dimension must match the DB column: products.embedding vector(768)
EMBEDDING_DIM = 768


async def embed_text(text: str) -> Optional[list[float]]:
    """
    Convert text to a float vector.

    Default: returns None (semantic search disabled → falls back to text search).

    To enable, implement using your provider:

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding

    Then set SEMANTIC_ENABLED = True above.
    """
    if not SEMANTIC_ENABLED:
        return None

    logger.warning("embed_text_called_but_not_implemented")
    return None


async def embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """Batch version — useful for backfilling embeddings."""
    if not SEMANTIC_ENABLED:
        return [None] * len(texts)

    results: list[Optional[list[float]]] = []
    for t in texts:
        results.append(await embed_text(t))
    return results