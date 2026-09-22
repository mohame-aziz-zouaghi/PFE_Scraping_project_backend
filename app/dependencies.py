"""Compatibility exports for applications importing shared dependencies here."""
from app.api.deps import get_current_admin, get_current_user, get_current_user_optional
from app.db.redis_client import get_redis
from app.db.session import get_db

__all__ = ["get_current_admin", "get_current_user", "get_current_user_optional", "get_db", "get_redis"]
