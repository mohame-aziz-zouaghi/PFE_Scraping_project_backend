from unittest.mock import AsyncMock, patch

import pytest

from app.services.embedding_service import EmbeddingService


@pytest.mark.asyncio
async def test_embed_batch_preserves_order():
    svc = EmbeddingService()
    texts = ["a", "b", "c"]
    fake_vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    with patch.object(
        svc,
        "_embed_provider",
        new=AsyncMock(return_value=(fake_vectors, "openai", "model", {})),
    ):
        result = await svc.embed_batch(texts)
    assert result == fake_vectors


@pytest.mark.asyncio
async def test_embed_batch_uses_cache_on_repeat():
    svc = EmbeddingService()
    texts = ["cached_text"]

    with patch.object(
        svc,
        "_embed_provider",
        new=AsyncMock(return_value=([[1.0, 0.0]], "openai", "model", {})),
    ) as mock_provider:
        await svc.embed_batch(texts)
        await svc.embed_batch(texts)
        assert mock_provider.call_count == 1
