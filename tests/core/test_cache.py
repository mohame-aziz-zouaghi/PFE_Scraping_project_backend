import pytest

from app.core.cache import Cache, make_key


def test_make_key_deterministic():
    a = make_key("llm", "recommend", 1, 2)
    b = make_key("llm", "recommend", 1, 2)
    assert a == b
    assert a.startswith("cache:llm:")


def test_make_key_differs():
    assert make_key("llm", "a") != make_key("llm", "b")


@pytest.mark.asyncio
async def test_cache_roundtrip():
    key = make_key("test", "roundtrip")
    await Cache.set(key, {"x": 1}, ttl_seconds=10)
    assert await Cache.get(key) == {"x": 1}


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    assert await Cache.get(make_key("test", "nonexistent")) is None


@pytest.mark.asyncio
async def test_get_or_set_calls_producer_once():
    calls = []

    async def producer():
        calls.append(1)
        return {"v": 42}

    key = make_key("test", "gos")
    r1 = await Cache.get_or_set(key, 10, producer)
    r2 = await Cache.get_or_set(key, 10, producer)
    assert r1 == r2 == {"v": 42}
    assert len(calls) == 1
