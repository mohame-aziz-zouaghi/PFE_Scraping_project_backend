from contextlib import asynccontextmanager
from pathlib import Path
import sys

from httpx import request

_THIS_FILE = Path(__file__).resolve()
_BACKEND_ROOT = _THIS_FILE.parent.parent
_PROJECT_ROOT = _BACKEND_ROOT.parent
for _p in (str(_BACKEND_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1.router import api_router
from app.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging_config import configure_logging, get_logger
from app.core.monitoring_adapter import monitoring_status
from app.db.redis_client import close_redis, get_redis, get_redis_pool_stats
from app.db.session import engine, dispose_engine, check_database_health, get_pool_stats
from app.db.redis_client import check_redis_health
from app.middleware.anomaly import AnomalyMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "app_starting",
        environment=settings.environment,
        version=settings.app_version,
    )

    # Redis
    try:
        app.state.redis = await get_redis()
        await app.state.redis.ping()
        logger.info("redis_connected")
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        app.state.redis = None

    logger.info("app_started")
    yield

    logger.info("app_shutting_down")
    try:
        await close_redis()
    except Exception as e:
        logger.warning("redis_close_error", error=str(e))
    try:
        await dispose_engine()
    except Exception as e:
        logger.warning("engine_dispose_error", error=str(e))
    logger.info("app_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AnomalyMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api/v1")

    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )

    @app.get("/health", include_in_schema=False)
    async def health():
        return {
            "status": "ok",
            "version": settings.app_version,
            "environment": settings.environment,
        }


    if settings.debug:
        @app.get("/pool-stats", include_in_schema=False)
        async def pool_stats():
            return {
                "database": get_pool_stats(),
                "redis": get_redis_pool_stats(),
            }

    @app.get("/ready", include_in_schema=False)
    async def ready(request: Request):
        redis_ok = await check_redis_health(request.app.state.redis)
        db_ok = await check_database_health(
            getattr(request.app.state, "db_engine", None)
        )
        monitor = monitoring_status()
        overall = "ready" if (redis_ok and db_ok) else "degraded"
        return {
            "status": overall,
            "checks": {
                "redis": redis_ok,
                "database": db_ok,
                "monitoring": {
                    "available": bool(monitor.get("available")),
                    "mode": monitor.get("mode"),
                    "reason": monitor.get("reason"),
                },
            },
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )