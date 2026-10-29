# app/api/v1/websocket.py
"""
WebSocket endpoints.

    /ws/progress          — monitoring progress (admin, live dashboard)
    /ws/alerts            — global alert counters (admin dashboard)
    /ws/user              — per-user notification push (auth via ?token=)

Wire paths (with router mounted at /api/v1):
    /api/v1/ws/progress
    /api/v1/ws/alerts
    /api/v1/ws/user
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
MAX_CONNECTIONS = 500


# ═══════════════════════════════════════════════════════
# Broadcasters
# ═══════════════════════════════════════════════════════
class ProgressBroadcaster:
    def __init__(self, max_connections: int = MAX_CONNECTIONS):
        self.connections: Set[WebSocket] = set()
        self.max_connections = max_connections
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> bool:
        async with self._lock:
            if len(self.connections) >= self.max_connections:
                return False
            await ws.accept()
            self.connections.add(ws)
            return True

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.connections.discard(ws)

    async def broadcast(self, message: dict):
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


class UserBroadcaster:
    def __init__(self, max_users: int = 10_000, max_sockets: int = 50_000):
        self._by_user: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self.max_users = max_users
        self.max_sockets = max_sockets
        self._socket_count = 0

    async def connect(self, user_id: str, ws: WebSocket) -> bool:
        async with self._lock:
            if len(self._by_user) >= self.max_users and user_id not in self._by_user:
                return False
            if self._socket_count >= self.max_sockets:
                return False
            await ws.accept()
            self._by_user.setdefault(user_id, set()).add(ws)
            self._socket_count += 1
            logger.info(
                "ws_user_broadcaster_connect",
                user_id=user_id,
                total_users=len(self._by_user),
                total_sockets=self._socket_count,
            )
            return True

    async def disconnect(self, user_id: str, ws: WebSocket):
        async with self._lock:
            bucket = self._by_user.get(user_id)
            if bucket is not None:
                bucket.discard(ws)
                self._socket_count = max(0, self._socket_count - 1)
                if not bucket:
                    self._by_user.pop(user_id, None)

    async def push(self, user_id: str, message: dict) -> int:
        async with self._lock:
            bucket = self._by_user.get(user_id)
            sockets = list(bucket) if bucket else []
        if not sockets:
            logger.info(
                "ws_user_broadcaster_no_sockets",
                user_id=user_id,
                total_users=len(self._by_user),
            )
            return 0

        payload = json.dumps(message, default=str)
        sent = 0
        dead: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_text(payload)
                sent += 1
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                bucket = self._by_user.get(user_id)
                if bucket is not None:
                    for ws in dead:
                        bucket.discard(ws)
                        self._socket_count = max(0, self._socket_count - 1)
                    if not bucket:
                        self._by_user.pop(user_id, None)

        logger.info(
            "ws_user_broadcaster_push",
            user_id=user_id,
            sockets_total=len(sockets),
            sockets_sent=sent,
        )
        return sent


broadcaster = ProgressBroadcaster()
user_broadcaster = UserBroadcaster()


# ═══════════════════════════════════════════════════════
# Public push API — called by alert_fanout
# ═══════════════════════════════════════════════════════
async def push_alert_to_user(user_id: str, alert_id: int) -> int:
    try:
        from app.db import acquire_db
        from sqlalchemy import text

        async with acquire_db(readonly=True) as db:
            row = (await db.execute(
                text("""
                    SELECT a.id, a.alert_type, a.old_value, a.new_value,
                           a.change_percent, a.site,
                           p.id::text AS product_id,
                           p.name AS product_name,
                           p.url AS product_url
                    FROM monitoring_alerts a
                    LEFT JOIN products p ON p.id = a.product_id
                    WHERE a.id = :aid
                """),
                {"aid": alert_id},
            )).first()

        if row is None:
            logger.warning(
                "push_alert_to_user_missing_alert",
                user_id=user_id, alert_id=alert_id,
            )
            return 0

        payload = {
            "type": "notification",
            "kind": (
                "price_alert" if row.alert_type == "price_drop"
                else "stock_alert"
            ),
            "alert_id": int(row.id),
            "title": _alert_title(row),
            "body": _alert_body(row),
            "product_id": row.product_id,
            "product_name": (row.product_name or "")[:200],
            "product_url": (row.product_url or "")[:500],
            "old_price": _num(row.old_value),
            "new_price": _num(row.new_value),
            "discount_pct": row.change_percent,
            "site": row.site,
        }
        return await user_broadcaster.push(user_id, payload)
    except Exception as exc:
        logger.warning(
            "push_alert_to_user_failed",
            user_id=user_id, alert_id=alert_id,
            error=str(exc)[:200],
        )
        return 0


async def push_unread_count(user_id: str) -> int:
    try:
        from app.db import acquire_db
        from sqlalchemy import text

        async with acquire_db(readonly=True) as db:
            n = (await db.execute(
                text(
                    "SELECT COUNT(*) FROM notifications "
                    "WHERE user_id = :uid AND is_read = FALSE"
                ),
                {"uid": user_id},
            )).scalar_one()

        return await user_broadcaster.push(
            user_id,
            {"type": "unread_count", "count": int(n)},
        )
    except Exception as exc:
        logger.debug(
            "push_unread_count_failed",
            user_id=user_id, error=str(exc)[:200],
        )
        return 0


def _alert_title(a) -> str:
    name = (a.product_name or "Product")[:80]
    if a.alert_type == "price_drop":
        return f"Price drop: {name}"
    if a.alert_type == "back_in_stock":
        return f"Back in stock: {name}"
    return f"Update: {name}"


def _alert_body(a) -> str:
    if a.alert_type == "price_drop":
        try:
            pct = float(a.change_percent or 0.0)
            return f"{a.old_value} → {a.new_value} ({pct:+.1f}%) on {a.site}"
        except Exception:
            return f"{a.old_value} → {a.new_value} on {a.site}"
    if a.alert_type == "back_in_stock":
        return f"Available again on {a.site}"
    return f"{a.old_value} → {a.new_value} on {a.site}"


def _num(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except Exception:
        return None


def _verify_ws_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    try:
        from app.core.security import decode_token
        payload = decode_token(token, expected_type="access")
        uid = payload.get("sub")
        return str(uid) if uid else None
    except Exception as exc:
        logger.warning("ws_token_invalid", error=str(exc)[:100])
        return None


# ═══════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════
async def _read_progress(redis: Redis, site: Optional[str]) -> dict:
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
        pass
    except Exception as e:
        logger.warning("ws_progress_error", error=str(e), site=site)
    finally:
        await broadcaster.disconnect(websocket)


@router.websocket("/ws/alerts")
async def ws_alerts(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    user_id = _verify_ws_token(token)
    if user_id is None:
        await websocket.close(code=4401, reason="Unauthorized")
        return

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


@router.websocket("/ws/user")
async def ws_user(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    user_id = _verify_ws_token(token)
    if user_id is None:
        logger.warning("ws_user_rejected_unauthorized")
        await websocket.close(code=4401, reason="Unauthorized")
        return

    if not await user_broadcaster.connect(user_id, websocket):
        await websocket.close(code=1013, reason="Server busy")
        return

    logger.info("ws_user_connected", user_id=user_id)

    try:
        await websocket.send_json({"type": "hello", "user_id": user_id})
        try:
            from app.db import acquire_db
            from sqlalchemy import text
            async with acquire_db(readonly=True) as db:
                n = (await db.execute(
                    text(
                        "SELECT COUNT(*) FROM notifications "
                        "WHERE user_id = :uid AND is_read = FALSE"
                    ),
                    {"uid": user_id},
                )).scalar_one()
            await websocket.send_json(
                {"type": "unread_count", "count": int(n)}
            )
        except Exception as exc:
            logger.debug(
                "ws_user_unread_initial_failed",
                user_id=user_id, error=str(exc)[:200],
            )

        while True:
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0
                )
                if msg == "ping":
                    await websocket.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(
            "ws_user_error", user_id=user_id, error=str(e)[:200]
        )
    finally:
        await user_broadcaster.disconnect(user_id, websocket)
        logger.info("ws_user_closed", user_id=user_id)