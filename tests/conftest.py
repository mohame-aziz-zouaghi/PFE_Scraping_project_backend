"""
Shared pytest fixtures.

- db_session: an AsyncSession wrapped in a transaction that rolls back
- client: an httpx AsyncClient wired to the FastAPI app
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_maker


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Session bound to a transaction that rolls back at the end."""
    async with async_session_maker() as session:
        tx = await session.begin()
        try:
            yield session
        finally:
            await tx.rollback()
            await session.close()


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    from app.main import create_app
    from contextlib import asynccontextmanager

    app = create_app()

    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    app.router.lifespan_context = _noop_lifespan
    app.state.redis = None  # no Redis in tests

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c



# tests/conftest.py
import pytest_asyncio
from app.db.session import engine, async_session_maker


@pytest_asyncio.fixture(autouse=True)
async def _reset_engine_between_tests():
    """
    Dispose the DB engine after every test so asyncpg connections
    aren't reused across event loops.

    Without this, pytest-asyncio creates a fresh event loop per test,
    but the asyncpg pool holds connections bound to the old loop.
    """
    yield
    # Dispose — closes all pooled connections bound to the (now dead) loop
    await engine.dispose()