"""
WebSocket endpoints — live monitoring progress.

Reads state from Redis (populated by the monitoring engine).
Falls back to DB when Redis has no live data.
"""
import asyncio
import json
from typing import Dict, Optional, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from redis.asyncio import Redis

from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

PROGRESS_KEY_PREFIX = "progress:site:"
POLL_INTERVAL = 1.0

# Hard cap per connection type to avoid abuse
MAX_CONNECTIONS = 500


class ProgressBroadcaster:
    """
    Connection manager with hard limits and per-site subscription.
    """

    def __init__(self, max_connections: int = MAX_CONNECTIONS):
        self.connections: Set[WebSocket] = set()
        self.max_connections = max_connections
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> bool:
        """
        Accept and register a connection.
        Returns False if the cap is reached (caller should reject).
        """
        async with self._lock:
            if len(self.connections) >= self.max_connections:
                logger.warning(
                    "ws_connection_cap_reached", cap=self.max_connections
                )
                return False
            await ws.accept()
            self.connections.add(ws)
            return True

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.connections.discard(ws)

    async def broadcast(self, message: dict):
        """Send to every connected client. Removes dead sockets."""
        if not self.connections:
            return
        payload = json.dumps(message, default=str)
        dead: list[WebSocket] = []
        async with self._lock:
            clients = list(self.connections)
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self.connections.discard(ws)


broadcaster = ProgressBroadcaster()


async def _read_progress(redis: Redis, site: Optional[str]) -> dict:
    """Read monitoring progress from Redis. Empty if nothing found."""
    result: dict = {}
    try:
        if site:
            sites = [site.lower()]
        else:
            sites = []
            cursor = 0
            pattern = f"{PROGRESS_KEY_PREFIX}*"
            while True:
                cursor, keys = await redis.scan(
                    cursor=cursor, match=pattern, count=100
                )
                for k in keys:
                    key = k.decode() if isinstance(k, bytes) else k
                    if key.endswith(":updated_at"):
                        continue
                    name = key[len(PROGRESS_KEY_PREFIX):]
                    if name and name not in sites:
                        sites.append(name)
                if cursor == 0:
                    break

        for s in sites:
            raw = await redis.get(f"{PROGRESS_KEY_PREFIX}{s}")
            if not raw:
                continue
            try:
                data = json.loads(raw)
                data["site"] = s
                result[s] = data
            except Exception:
                continue
    except Exception as e:
        logger.warning("progress_read_failed", error=str(e))
    return result


@router.websocket("/ws/progress")
async def ws_progress(
    websocket: WebSocket,
    site: Optional[str] = Query(None, max_length=50),
    token: Optional[str] = Query(None),
):
    """
    Stream monitoring progress.

    Usage:
        ws://host/api/v1/ws/progress?site=megapc
        ws://host/api/v1/ws/progress (all sites)
    """
    if not await broadcaster.connect(websocket):
        await websocket.close(code=1013, reason="Server busy")
        return

    redis: Optional[Redis] = getattr(websocket.app.state, "redis", None)

    try:
        await websocket.send_json(
            {"type": "hello", "message": "connected", "site": site}
        )
        while True:
            if redis is not None:
                data = await _read_progress(redis, site)
                await websocket.send_json({"type": "progress", "sites": data})
            else:
                await websocket.send_json({"type": "progress", "sites": {}})
            await asyncio.sleep(POLL_INTERVAL)
    except WebSocketDisconnect:
        logger.debug("ws_progress_disconnected", site=site)
    except Exception as e:
        logger.warning("ws_progress_error", error=str(e), site=site)
    finally:
        await broadcaster.disconnect(websocket)


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """
    Stream alerts count for live badges.
    Reads lightweight counters from Redis.
    """
    if not await broadcaster.connect(websocket):
        await websocket.close(code=1013, reason="Server busy")
        return

    redis: Optional[Redis] = getattr(websocket.app.state, "redis", None)

    try:
        await websocket.send_json({"type": "hello", "message": "connected"})
        while True:
            payload = {"unresolved": 0, "recent": 0}
            if redis is not None:
                try:
                    unresolved = await redis.get("alerts:unresolved_count")
                    recent = await redis.get("alerts:recent_count")
                    payload["unresolved"] = int(unresolved) if unresolved else 0
                    payload["recent"] = int(recent) if recent else 0
                except Exception:
                    pass
            await websocket.send_json({"type": "alerts", "data": payload})
            await asyncio.sleep(POLL_INTERVAL * 2)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("ws_alerts_error", error=str(e))
    finally:
        await broadcaster.disconnect(websocket)