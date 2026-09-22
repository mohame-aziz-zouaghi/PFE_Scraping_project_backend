"""Tests for EmbeddingRepo — require a live DB with pgvector."""
from uuid import uuid4

import pytest

from app.repositories.embedding_repo import EmbeddingRepo

pytestmark = pytest.mark.asyncio


async def test_get_by_product_returns_none_for_missing(db_session):
    repo = EmbeddingRepo(db_session)
    result = await repo.get_by_product(uuid4(), "test-model")
    assert result is None


async def test_count_ready_returns_int(db_session):
    repo = EmbeddingRepo(db_session)
    count = await repo.count_ready()
    assert isinstance(count, int)
    assert count >= 0


async def test_raw_stats_returns_dict(db_session):
    repo = EmbeddingRepo(db_session)
    stats = await repo.raw_stats()
    assert isinstance(stats, dict)