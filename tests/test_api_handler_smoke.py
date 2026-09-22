"""HTTP smoke coverage for every database-backed public API route.

Services are replaced at their dependency boundary; this validates FastAPI
binding, request validation, auth overrides, response serialization and route
status without requiring PostgreSQL, Redis, or a scraper installation.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

import app.api.v1.alerts as alerts_module
import app.api.v1.analytics as analytics_module
import app.api.v1.monitoring as monitoring_module
import app.api.v1.products as products_module
import app.api.v1.search as search_module
from app.api.deps import get_current_admin, get_current_user, get_current_user_optional
from app.core.security import create_refresh_token, hash_password
from app.db.session import get_db
from app.main import create_app
from app.models.user import User


NOW = datetime.now(timezone.utc)
PRODUCT_ID = uuid4()
USER_ID = uuid4()


def product():
    return {"id": str(PRODUCT_ID), "site": "megapc", "name": "GPU", "brand": "A", "category": "GPU", "price": 100.0, "discount_price": None, "availability": "InStock", "main_image": None, "last_updated_at": NOW.isoformat(), "main_product_id": "p1", "url": "https://example.test/p1", "price_valid_until": None, "is_enabled": True, "last_checked": None}


class ProductServiceStub:
    def __init__(self, db): pass
    async def list_products(self, **kwargs): return {"items": [product()], "total": 1, "page": 1, "page_size": 20, "total_pages": 1}
    async def get_product(self, product_id): return product()
    async def get_price_history(self, product_id, page, page_size): return {"product_id": str(product_id), "items": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 0}
    async def get_change_history(self, product_id, page, page_size): return {"product_id": str(product_id), "items": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 0}


class SearchServiceStub:
    def __init__(self, db): pass
    async def text_search(self, payload): return {"query": payload.query, "items": [], "total": 0, "page": payload.page, "page_size": payload.page_size, "total_pages": 0, "took_ms": 0}
    async def suggestions(self, query, limit): return {"query": query, "items": []}
    async def semantic_search(self, payload): return {"query": payload.query, "items": [], "total": 0, "page": 1, "page_size": payload.limit, "total_pages": 0, "took_ms": 0}


class AnalyticsServiceStub:
    def __init__(self, db): pass
    async def get_price_trends(self, **kwargs): return {"product_id": None, "site": kwargs["site"], "period_days": kwargs["period_days"], "points": []}
    async def get_volatile_products(self, **kwargs): return {"period_days": kwargs["period_days"], "items": [], "total": 0}
    async def get_price_drops(self, **kwargs): return {"period_days": kwargs["period_days"], "items": [], "total": 0}
    async def get_alert_summary(self, **kwargs): return {"total": 0, "unresolved": 0, "resolved": 0, "by_site": {}, "by_type": {}, "by_severity": {}}
    async def get_availability_timeline(self, **kwargs): return {"site": kwargs["site"], "period_days": kwargs["period_days"], "points": []}
    async def get_sites_overview(self): return {"items": [], "total_sites": 0}


class AlertServiceStub:
    def __init__(self, db): pass
    async def list_alerts(self, **kwargs): return {"items": [], "total": 0, "page": kwargs["page"], "page_size": kwargs["page_size"], "total_pages": 0}
    async def resolve(self, alert_id): return {"message": "Alert resolved", "alert": {"id": alert_id, "product_id": str(PRODUCT_ID), "site": "megapc", "alert_type": "price_drop", "old_value": 120, "new_value": 100, "severity": "info", "change_percent": -16.7, "is_resolved": True, "resolved_at": NOW.isoformat(), "created_at": NOW.isoformat()}}


def build_client(monkeypatch):
    monkeypatch.setattr(products_module, "ProductService", ProductServiceStub)
    monkeypatch.setattr(search_module, "SearchService", SearchServiceStub)
    monkeypatch.setattr(analytics_module, "AnalyticsService", AnalyticsServiceStub)
    monkeypatch.setattr(alerts_module, "AlertService", AlertServiceStub)
    app = create_app()
    @asynccontextmanager
    async def no_external_startup(app):
        yield
    app.router.lifespan_context = no_external_startup
    user = SimpleNamespace(id=USER_ID, tier="admin", is_active=True)
    async def db_override(): yield SimpleNamespace()
    async def current_user(): return user
    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_current_user] = current_user
    app.dependency_overrides[get_current_admin] = current_user
    app.dependency_overrides[get_current_user_optional] = current_user
    return TestClient(app)


def test_product_search_analytics_and_alert_routes(monkeypatch):
    with build_client(monkeypatch) as client:
        responses = [
            client.get("/api/v1/products"), client.get(f"/api/v1/products/{PRODUCT_ID}"),
            client.get(f"/api/v1/products/{PRODUCT_ID}/history"), client.get(f"/api/v1/products/{PRODUCT_ID}/changes"),
            client.post("/api/v1/search/text", json={"query": "gpu"}), client.get("/api/v1/search/quick?q=gpu"),
            client.get("/api/v1/search/suggestions?q=g"), client.post("/api/v1/search/semantic", json={"query": "gpu"}),
            client.get("/api/v1/analytics/price-trends"), client.get("/api/v1/analytics/volatile-products"),
            client.get("/api/v1/analytics/price-drops"), client.get("/api/v1/analytics/alerts-summary"),
            client.get("/api/v1/analytics/availability-timeline"), client.get("/api/v1/analytics/sites-overview"),
            client.get("/api/v1/alerts"), client.post("/api/v1/alerts/1/resolve"),
        ]
    assert [response.status_code for response in responses] == [200] * len(responses)


def test_monitoring_status_and_unavailable_trigger(monkeypatch):
    monkeypatch.setattr(monitoring_module, "monitoring_status", lambda: {"available": False, "reason": "adapter absent", "sites": []})
    with build_client(monkeypatch) as client:
        status = client.get("/api/v1/monitoring/status")
        trigger = client.post("/api/v1/monitoring/sites/megapc/trigger")
    assert status.status_code == 200
    assert trigger.status_code == 503
    assert trigger.json()["error"]["code"] == "monitoring_unavailable"


class Result:
    def __init__(self, value=None, rows=None): self.value, self.rows = value, rows or []
    def scalar_one_or_none(self): return self.value
    def scalar_one(self): return self.value
    def scalars(self): return SimpleNamespace(all=lambda: self.rows)


class AuthDb:
    def __init__(self, user=None): self.user = user
    async def execute(self, statement): return Result(self.user)
    def add(self, entity):
        if isinstance(entity, User):
            entity.id, entity.created_at = uuid4(), NOW
            entity.is_active, entity.is_verified = True, False
    async def flush(self): pass
    async def commit(self): pass
    async def refresh(self, entity): pass


def test_auth_routes_execute_successfully_with_session_boundary():
    app = create_app()
    @asynccontextmanager
    async def no_external_startup(app): yield
    app.router.lifespan_context = no_external_startup
    user = User(id=USER_ID, email="admin@example.com", username="admin", hashed_password=hash_password("strong-password"), tier="admin", is_active=True, is_verified=True, created_at=NOW)
    db = AuthDb()
    async def db_override(): yield db
    async def current_user(): return user
    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_current_user] = current_user
    with TestClient(app) as client:
        registered = client.post("/api/v1/auth/register", json={"email": "new@example.com", "username": "newuser", "password": "strong-password"})
        db.user = user
        login = client.post("/api/v1/auth/login", json={"email": "admin@example.com", "password": "strong-password"})
        client.cookies.set("refresh_token", create_refresh_token(str(USER_ID)), path="/api/v1/auth/refresh")
        refreshed = client.post("/api/v1/auth/refresh")
        me = client.get("/api/v1/auth/me")
        logout = client.post("/api/v1/auth/logout")
    assert [response.status_code for response in (registered, login, refreshed, me, logout)] == [201, 200, 200, 200, 200]


def test_monitoring_batch_and_successful_trigger(monkeypatch):
    app = create_app()
    @asynccontextmanager
    async def no_external_startup(app): yield
    app.router.lifespan_context = no_external_startup
    row = SimpleNamespace(id=uuid4(), site="megapc", batch_type="scheduled", status="completed", total_products=1, processed_products=1, changed_products=0, error_products=0, start_time=NOW, end_time=NOW, duration_seconds=1)
    class BatchDb:
        def __init__(self): self.calls = 0
        async def execute(self, statement):
            self.calls += 1
            return Result(1) if self.calls == 1 else Result(rows=[row])
    async def db_override(): yield BatchDb()
    async def admin(): return SimpleNamespace(id=USER_ID, tier="admin")
    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_current_user] = admin
    app.dependency_overrides[get_current_admin] = admin
    monkeypatch.setattr(monitoring_module, "monitoring_status", lambda: {"available": True, "reason": None, "sites": ["megapc"]})
    monkeypatch.setattr(monitoring_module, "run_site_monitoring", SimpleNamespace(delay=lambda site: SimpleNamespace(id="task-1")))
    with TestClient(app) as client:
        batches = client.get("/api/v1/monitoring/batches")
        trigger = client.post("/api/v1/monitoring/sites/megapc/trigger")
    assert batches.status_code == 200
    assert trigger.json() == {"message": "Triggered", "task_id": "task-1" , "site": "megapc"}
