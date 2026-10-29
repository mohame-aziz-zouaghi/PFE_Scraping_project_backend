"""
Persistent event loop for Celery async tasks.

Celery workers are synchronous. To run async DB code from them, you
need an event loop. Using `asyncio.run(coro)` per task creates and
destroys a loop every time, forcing the asyncpg pool to be disposed
between tasks (defeating the pool entirely).

This module provides ONE event loop per worker process. All tasks
submit their coroutines to this loop. DB connections created inside
it survive across tasks, so the pool actually pools.

Lifecycle:
    - Loop is created lazily on first task.
    - Loop runs until the worker process exits.
    - On worker shutdown, the loop is stopped and drained.
    - Engine dispose happens once, at worker shutdown.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()
_shutdown_requested = threading.Event()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create the persistent event loop for this process.
    Thread-safe: concurrent calls yield the same loop.
    """
    global _loop, _thread

    with _lock:
        if _loop is not None and _loop.is_running():
            return _loop

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _loop = loop

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_forever()
            finally:
                try:
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        logger.info(
                            "async_runner_draining", pending=len(pending)
                        )
                        for task in pending:
                            task.cancel()
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception as exc:
                    logger.warning(
                        "async_runner_drain_failed", error=str(exc)[:200]
                    )
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                loop.close()

        thread = threading.Thread(
            target=_runner,
            name="celery-async-runner",
            daemon=True,
        )
        thread.start()
        _thread = thread

        logger.info(
            "async_runner_started",
            loop_id=id(loop),
            thread=thread.name,
        )
        return loop


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run a coroutine on the persistent event loop and block the
    calling (sync) thread until it completes.
    """
    loop = _ensure_loop()

    if _shutdown_requested.is_set():
        raise RuntimeError("async runner is shutting down")

    future: Future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result()
    except KeyboardInterrupt:
        future.cancel()
        raise


def request_shutdown() -> None:
    """
    Signal the persistent loop to stop. Called from Celery's
    worker_shutdown signal.
    """
    global _loop, _thread
    _shutdown_requested.set()

    with _lock:
        loop = _loop
        thread = _thread
        _loop = None
        _thread = None

    if loop is None:
        return

    logger.info("async_runner_shutdown_requested")
    try:
        loop.call_soon_threadsafe(loop.stop)
    except Exception as exc:
        logger.warning(
            "async_runner_stop_failed", error=str(exc)[:200]
        )
    if thread is not None and thread.is_alive():
        thread.join(timeout=10.0)


atexit.register(request_shutdown)