"""
Async SQLAlchemy session factory.

Works in three environments:
- Local dev (no Docker): connects to localhost
- Docker: connects to service names (postgres, redis)
- Remote: connects to managed services via DATABASE_URL

Uses settings for all pool tuning.
"""
import logging
from contextlib import asynccontextmanager           # ← NEW
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool

from app.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────
_db_url = settings.async_database_url

engine = create_async_engine(
    _db_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_timeout=settings.database_pool_timeout,
    pool_recycle=settings.database_pool_recycle,
    pool_pre_ping=True,
    pool_use_lifo=True,
    echo=False,
    future=True,
)


# ─────────────────────────────────────────────────────
# Pool diagnostics (optional — great for debugging)
# ─────────────────────────────────────────────────────
@event.listens_for(engine.sync_engine, "connect")
def _on_connect(dbapi_conn, connection_record):
    logger.debug("db_pool | connection_opened")


@event.listens_for(engine.sync_engine, "checkout")
def _on_checkout(dbapi_conn, connection_record, connection_proxy):
    pool = engine.pool
    logger.debug(
        "db_pool | connection_checkout | size=%d | in_use=%d | overflow=%d",
        pool.size(),
        pool.checkedout(),
        pool.overflow(),
    )


@event.listens_for(engine.sync_engine, "checkin")
def _on_checkin(dbapi_conn, connection_record):
    logger.debug("db_pool | connection_checkin")


@event.listens_for(engine.sync_engine, "invalidate")
def _on_invalidate(dbapi_conn, connection_record, exception):
    logger.warning("db_pool | connection_invalidated | error=%s", exception)


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

# Public alias — scripts and Celery tasks use this name.
# Kept as a real binding (not an import-time alias) so tooling can find it.
async_session_maker = AsyncSessionLocal              # ← NEW


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — one session per request.

    Lifecycle:
    - Opens a session when the request enters.
    - Rolls back on unhandled exception.
    - Always closes the session in finally.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            try:
                await session.rollback()
            except Exception as rb_err:
                logger.error("db_session_rollback_failed | %s", rb_err)
            raise
        finally:
            try:
                await session.close()
            except Exception as close_err:
                logger.error("db_session_close_failed | %s", close_err)


@asynccontextmanager                               # ← NEW
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for scripts, Celery tasks, and one-off jobs.

    Commits on clean exit, rolls back on exception, always closes.
    Usage:
        async with session_scope() as db:
            db.add(obj)
            # auto-commits on exit
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            try:
                await session.rollback()
            except Exception as rb_err:
                logger.error("db_session_scope_rollback_failed | %s", rb_err)
            raise
        finally:
            try:
                await session.close()
            except Exception as close_err:
                logger.error("db_session_scope_close_failed | %s", close_err)


async def check_database_health(engine_override=None) -> bool:
    eng = engine_override or engine
    try:
        async with eng.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
        return True
    except Exception as e:
        logger.warning(f"db_health_check_failed | {e}")
        return False


def get_pool_stats() -> dict:
    """Return pool stats — helpful for /metrics or /ready diagnostics."""
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "available": pool.size() - pool.checkedout(),
    }


async def dispose_engine() -> None:
    """Close all pool connections. Called on shutdown."""
    try:
        await engine.dispose()
        logger.info("db_engine_disposed")
    except Exception as e:
        logger.warning("db_engine_dispose_failed | %s", e)