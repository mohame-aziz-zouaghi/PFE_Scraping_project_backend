"""
Anomaly detection: block banned IPs, detect unusual request patterns.
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.honeypot import is_ip_banned
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class AnomalyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        redis = getattr(request.app.state, "redis", None)
        if redis is None:
            return await call_next(request)
        
        ip = request.client.host if request.client else "unknown"
        
        # Check ban list
        try:
            if await is_ip_banned(redis, ip):
                logger.warning("banned_ip_blocked", ip=ip, path=request.url.path)
                return JSONResponse(
                    status_code=403,
                    content={"error": {"code": "forbidden", "message": "Access denied"}},
                )
        except Exception:
            # Redis down — fail open (don't block legit traffic)
            pass
        
        return await call_next(request)