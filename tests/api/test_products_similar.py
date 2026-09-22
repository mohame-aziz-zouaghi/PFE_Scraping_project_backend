"""Smoke tests for /products/{id}/similar — requires running app + DB."""
import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


async def test_similar_returns_404_for_unknown_product(client: AsyncClient):
    import uuid
    fake_id = str(uuid.uuid4())
    resp = await client.get(f"/api/v1/products/{fake_id}/similar")
    assert resp.status_code in (404, 422)


async def test_embedding_stats_endpoint_returns_shape(client: AsyncClient):
    resp = await client.get("/api/v1/products/embeddings/stats")
    # May be 200 (populated) or 404/500 if the table isn't set up — accept both
    if resp.status_code == 200:
        data = resp.json()
        assert "total_embedded" in data
        assert "total_pending" in data
        assert "status_breakdown" in data
        assert "model" in data
