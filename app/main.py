from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import sys
import traceback

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
from app.db.session import (
    engine, dispose_engine, check_database_health, get_pool_stats,
    start_pool_reaper, stop_pool_reaper,
)
from app.db.redis_client import check_redis_health
from app.middleware.anomaly import AnomalyMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.middleware.activity import ActivityMiddleware

from app.services.embedding_queue import embedding_queue
from app.services.help_article_hooks import register_help_article_hooks
from app.services.product_hooks import register_product_hooks
from app.db.db_listener import pg_listener
from app.services.alert_fanout import alert_fanout

configure_logging()
logger = get_logger(__name__)

REDIS_INIT_TIMEOUT_S = 5.0
DB_INIT_TIMEOUT_S = 5.0
RAG_INIT_TIMEOUT_S = 10.0
PREWARM_DELAY_S = 2.0


async def _with_timeout(coro, timeout_s: float, name: str):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.error(f"{name}_timeout", timeout_s=timeout_s)
        return None
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error(
            f"{name}_failed",
            error=str(exc)[:300],
            error_type=type(exc).__name__,
            traceback=tb[-3000:],
        )
        return None


async def _init_redis(app: FastAPI) -> None:
    try:
        client = await asyncio.wait_for(get_redis(), timeout=REDIS_INIT_TIMEOUT_S)
        if client is None:
            raise RuntimeError("get_redis() returned None")
        app.state.redis = client
        logger.info("redis_connected")
    except asyncio.TimeoutError:
        logger.error("redis_init_timeout", timeout_s=REDIS_INIT_TIMEOUT_S)
        app.state.redis = None
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error(
            "redis_init_failed",
            error=str(exc)[:300],
            error_type=type(exc).__name__,
            traceback=tb[-3000:],
        )
        app.state.redis = None


async def _init_db_pool() -> bool:
    try:
        start_pool_reaper()
        logger.info("db_pool_reaper_started")
        ok = await asyncio.wait_for(
            check_database_health(), timeout=DB_INIT_TIMEOUT_S,
        )
        if not ok:
            raise RuntimeError("check_database_health returned False")
        logger.info("db_pool_ready")
        return True
    except asyncio.TimeoutError:
        logger.error("db_pool_init_timeout", timeout_s=DB_INIT_TIMEOUT_S)
        return False
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error(
            "db_pool_init_failed",
            error=str(exc)[:300],
            error_type=type(exc).__name__,
            traceback=tb[-3000:],
        )
        return False


async def _rag_startup() -> None:
    try:
        register_help_article_hooks()
        register_product_hooks()
    except Exception as exc:
        logger.warning("rag_hooks_register_failed", error=str(exc)[:200])
    try:
        embedding_queue.start()
    except Exception as exc:
        logger.warning("rag_queue_start_failed", error=str(exc)[:200])
    try:
        pg_listener.start()
    except Exception as exc:
        logger.warning("pg_listener_start_failed", error=str(exc)[:200])
    async def _reconcile():
        counts = await embedding_queue.reconcile_all()
        logger.info("rag_reconcile_done", **counts)
    await _with_timeout(_reconcile(), RAG_INIT_TIMEOUT_S, "rag_reconcile")


async def _rag_prewarm_bg() -> None:
    await asyncio.sleep(PREWARM_DELAY_S)
    try:
        from app.services.rag_cache import prewarm_anon_suggestions
        await asyncio.wait_for(
            prewarm_anon_suggestions(locale="en", top_n=10), timeout=15.0,
        )
    except asyncio.TimeoutError:
        logger.warning("rag_anon_prewarm_timeout")
    except Exception as exc:
        logger.warning("rag_anon_prewarm_failed", error=str(exc)[:200])


def _start_alert_fanout() -> None:
    try:
        alert_fanout.start()
        logger.info("alert_fanout_started")
    except Exception as exc:
        logger.warning("alert_fanout_start_failed", error=str(exc)[:200])


async def _stop_alert_fanout() -> None:
    try:
        await asyncio.wait_for(alert_fanout.stop(), timeout=10.0)
        logger.info("alert_fanout_stopped")
    except asyncio.TimeoutError:
        logger.warning("alert_fanout_stop_timeout")
    except Exception as exc:
        logger.warning("alert_fanout_stop_failed", error=str(exc)[:200])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_starting", environment=settings.environment,
                version=settings.app_version)

    await _init_redis(app)
    await _init_db_pool()
    await _rag_startup()
    prewarm_task = asyncio.create_task(_rag_prewarm_bg())
    _start_alert_fanout()

    logger.info("app_started")
    yield

    logger.info("app_shutting_down")

    if not prewarm_task.done():
        prewarm_task.cancel()
        try:
            await prewarm_task
        except (asyncio.CancelledError, Exception):
            pass

    await _stop_alert_fanout()

    try:
        await asyncio.wait_for(pg_listener.stop(), timeout=5.0)
    except Exception as exc:
        logger.warning("pg_listener_stop_error", error=str(exc)[:200])

    try:
        await asyncio.wait_for(embedding_queue.stop(), timeout=10.0)
    except Exception as exc:
        logger.warning("rag_queue_stop_error", error=str(exc)[:200])

    try:
        await asyncio.wait_for(stop_pool_reaper(), timeout=5.0)
    except Exception as exc:
        logger.warning("db_pool_reaper_stop_error", error=str(exc))

    try:
        await asyncio.wait_for(close_redis(), timeout=5.0)
    except Exception as exc:
        logger.warning("redis_close_error", error=str(exc))

    try:
        await asyncio.wait_for(dispose_engine(), timeout=5.0)
    except Exception as exc:
        logger.warning("engine_dispose_error", error=str(exc))

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

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ActivityMiddleware)
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

    @app.get("/metrics/pools", include_in_schema=False)
    async def metrics_pools():
        return {
            "database": get_pool_stats(),
            "redis": get_redis_pool_stats(),
        }

    @app.get("/ready", include_in_schema=False)
    async def ready(request: Request):
        redis_ok = await check_redis_health(request.app.state.redis)
        db_ok = await check_database_health()
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
            "pools": {
                "database": get_pool_stats(),
                "redis": get_redis_pool_stats(),
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