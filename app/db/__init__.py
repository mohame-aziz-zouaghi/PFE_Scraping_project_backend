"""
Public DB + Redis surface.

    from app.db import acquire_db, acquire_redis

Existing imports (from app.db.session / app.db.redis_client) still work.
"""
from app.db.session import (
    acquire_db,
    acquire_db_reentrant,
    async_session_maker,
    check_database_health,
    db_session_dependency,
    dispose_engine,
    engine,
    get_db,
    get_pool_stats,
    healthcheck as db_healthcheck,
    session_scope,
    with_db,
)
from app.db.redis_client import (
    RedisLease,
    acquire_redis,
    cache_delete,
    cache_get,
    cache_set,
    check_redis_health,
    close_redis,
    get_redis,
    get_redis_pool_stats,
    with_redis,
)

__all__ = [
    "acquire_db", "acquire_db_reentrant", "async_session_maker",
    "check_database_health", "db_session_dependency", "dispose_engine",
    "engine", "get_db", "get_pool_stats", "db_healthcheck",
    "session_scope", "with_db",
    "RedisLease", "acquire_redis", "cache_delete", "cache_get",
    "cache_set", "check_redis_health", "close_redis", "get_redis",
    "get_redis_pool_stats", "with_redis",
]