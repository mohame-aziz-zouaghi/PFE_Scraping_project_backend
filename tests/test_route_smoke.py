"""Dependency-isolated smoke tests for infrastructure and adapter routes."""
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import create_app


class FakeRedis:
    def __init__(self):
        self._data = {}

    async def ping(self):
        return True

    async def get(self, key):
        return self._data.get(key)

    async def set(self, key, value, ex=None):
        self._data[key] = value

    async def delete(self, key):
        self._data.pop(key, None)

    async def incr(self, key):
        self._data[key] = int(self._data.get(key, 0)) + 1
        return self._data[key]

    async def expire(self, key, seconds):
        return True


class FakeConnection:
    async def exec_driver_sql(self, statement):
        return None


class FakeEngine:
    @asynccontextmanager
    async def connect(self):
        yield FakeConnection()


def test_health_and_readiness_return_structured_status(monkeypatch):
    app = create_app()
    # Avoid external service startup while retaining the actual request pipeline.
    async def lifespan_override(app):
        yield
    app.router.lifespan_context = asynccontextmanager(lifespan_override)
    monkeypatch.setattr(main_module, "engine", FakeEngine())
    with TestClient(app) as client:
        app.state.redis = FakeRedis()
        health = client.get("/health")
        ready = client.get("/ready")
        metrics = client.get("/metrics")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert metrics.status_code == 200


def test_honeypot_returns_not_found_when_redis_is_available():
    app = create_app()
    async def lifespan_override(app):
        yield
    app.router.lifespan_context = asynccontextmanager(lifespan_override)
    with TestClient(app) as client:
        app.state.redis = FakeRedis()
        responses = [client.request(method, path) for method, path in (
            ("GET", "/api/v1/.env"), ("GET", "/api/v1/backup.sql"),
            ("GET", "/api/v1/wp-admin"), ("GET", "/api/v1/.git/config"),
            ("GET", "/api/v1/internal/admin/users"), ("POST", "/api/v1/internal/admin/users"),
        )]
    assert [response.status_code for response in responses] == [404] * len(responses)
    assert all(response.json()["error"]["code"] == "not_found" for response in responses)


def test_websocket_routes_connect_and_emit_initial_payloads():
    app = create_app()
    async def lifespan_override(app):
        yield
    app.router.lifespan_context = asynccontextmanager(lifespan_override)
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/progress") as socket:
            assert socket.receive_json()["type"] == "hello"
            assert socket.receive_json()["type"] == "progress"
        with client.websocket_connect("/api/v1/ws/alerts") as socket:
            assert socket.receive_json()["type"] == "hello"
            assert socket.receive_json()["type"] == "alerts"
