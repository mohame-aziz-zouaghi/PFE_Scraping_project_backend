"""
Async SQLAlchemy session factory + self-releasing leases.

Pool behaviour:
    - Size grows 5 -> 30 on demand.
    - Idle connections evicted after 5s (down to min=5).
    - Statement timeout 45s, idle-in-tx timeout 0 (disabled) per connection.

Backward-compatible: every symbol the old file exported is still here.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, TypeVar

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.db.dynamic_pool import DynamicAsyncPool
from app.db.pool_config import POOL

logger = logging.getLogger(__name__)

T = TypeVar("T")

POOL_STATS: dict[str, int] = {
    "checkouts": 0,
    "checkins": 0,
    "invalidations": 0,
}


# ─────────────────────────────────────────────────────
# Engine — ONLY poolclass goes here. No pool kwargs.
# ─────────────────────────────────────────────────────
_db_url = settings.async_database_url

engine: AsyncEngine = create_async_engine(
    _db_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_use_lifo=True,
    poolclass=DynamicAsyncPool,
    connect_args={
        "server_settings": {
            "application_name": "price_platform",
            "statement_timeout": str(POOL.db_statement_timeout_ms),
            "idle_in_transaction_session_timeout":
                str(POOL.db_idle_in_tx_timeout_ms),
        },
    },
)


# ─────────────────────────────────────────────────────
# Pool event hooks
# ─────────────────────────────────────────────────────
@event.listens_for(engine.sync_engine, "checkout")
def _on_checkout(dbapi_conn, connection_record, connection_proxy):
    POOL_STATS["checkouts"] += 1
    pool = engine.pool
    logger.debug(
        "db_pool | checkout | in_use=%d idle=%d",
        pool.checkedout(),
        pool.dynamic_stats()["idle"] if hasattr(pool, "dynamic_stats") else -1,
    )


@event.listens_for(engine.sync_engine, "checkin")
def _on_checkin(dbapi_conn, connection_record):
    POOL_STATS["checkins"] += 1


@event.listens_for(engine.sync_engine, "invalidate")
def _on_invalidate(dbapi_conn, connection_record, exception):
    POOL_STATS["invalidations"] += 1
    logger.warning("db_pool | invalidated | error=%s", exception)


# ─────────────────────────────────────────────────────
# Session factory
# ─────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)
async_session_maker = AsyncSessionLocal


# ─────────────────────────────────────────────────────
# Lease
# ─────────────────────────────────────────────────────
_ctx_db: contextvars.ContextVar[Optional[AsyncSession]] = contextvars.ContextVar(
    "current_db_session", default=None
)


async def _safe_rollback(session: AsyncSession) -> None:
    try:
        await session.rollback()
    except Exception as e:
        logger.error("db_rollback_failed | %s", e)


async def _safe_close(session: AsyncSession) -> None:
    try:
        await session.close()
    except Exception as e:
        logger.error("db_close_failed | %s", e)


@asynccontextmanager
async def acquire_db(
    *,
    auto_commit: bool = True,
    readonly: bool = False,
) -> AsyncGenerator[AsyncSession, None]:
    """Lease a session. Always releases the connection on exit."""
    t0 = time.monotonic()
    session = AsyncSessionLocal()
    try:
        yield session
        if auto_commit and not readonly:
            await session.commit()
    except asyncio.CancelledError:
        await _safe_rollback(session)
        raise
    except Exception:
        await _safe_rollback(session)
        raise
    finally:
        await _safe_close(session)
        duration_ms = (time.monotonic() - t0) * 1000
        if duration_ms > POOL.db_lease_warn_ms:
            logger.warning(
                "db_lease_slow | duration_ms=%.1f", duration_ms,
            )


@asynccontextmanager
async def acquire_db_reentrant(
    *,
    auto_commit: bool = True,
    readonly: bool = False,
) -> AsyncGenerator[AsyncSession, None]:
    outer = _ctx_db.get()
    if outer is not None:
        yield outer
        return
    async with acquire_db(auto_commit=auto_commit, readonly=readonly) as db:
        token = _ctx_db.set(db)
        try:
            yield db
        finally:
            _ctx_db.reset(token)


# ─────────────────────────────────────────────────────
# Backward-compatible exports
# ─────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with acquire_db() as session:
        yield session


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    async with acquire_db() as session:
        yield session


def with_db(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        async with acquire_db() as db:
            return await fn(*args, db=db, **kwargs)
    return wrapper


async def db_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    async with acquire_db() as db:
        yield db


# ─────────────────────────────────────────────────────
# Reaper lifecycle
# ─────────────────────────────────────────────────────
def start_pool_reaper() -> None:
    pool = engine.pool
    if isinstance(pool, DynamicAsyncPool):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        pool.start_reaper(loop)


async def stop_pool_reaper() -> None:
    pool = engine.pool
    if isinstance(pool, DynamicAsyncPool):
        await pool.stop_reaper()


# ─────────────────────────────────────────────────────
# Health / stats / lifecycle
# ─────────────────────────────────────────────────────
async def check_database_health(engine_override=None) -> bool:
    eng = engine_override or engine
    try:
        async with eng.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
        return True
    except Exception as e:
        logger.warning("db_health_check_failed | %s", e)
        return False


async def healthcheck() -> tuple[bool, str]:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as e:
        return False, repr(e)


def get_pool_stats() -> dict:
    pool = engine.pool
    base = {
        "size":        pool.size(),
        "checked_out": pool.checkedout(),
        "overflow":    pool.overflow(),
    }
    if isinstance(pool, DynamicAsyncPool):
        base.update(pool.dynamic_stats())
    base.update(POOL_STATS)
    return base


async def dispose_engine() -> None:
    try:
        await stop_pool_reaper()
    except Exception:
        pass
    try:
        await engine.dispose()
        logger.info("db_engine_disposed")
    except Exception as e:
        logger.warning("db_engine_dispose_failed | %s", e)