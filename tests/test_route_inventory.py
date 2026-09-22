"""Route inventory checks: every public endpoint must remain registered."""
import logging

from fastapi.routing import APIRoute, APIWebSocketRoute

from app.main import create_app

logger = logging.getLogger(__name__)

HTTP_ROUTES = {
    ("POST", "/api/v1/auth/register"), ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/refresh"), ("POST", "/api/v1/auth/logout"),
    ("GET", "/api/v1/auth/me"),
    ("GET", "/api/v1/products"), ("GET", "/api/v1/products/{product_id}"),
    ("GET", "/api/v1/products/{product_id}/history"), ("GET", "/api/v1/products/{product_id}/changes"),
    ("GET", "/api/v1/alerts"), ("POST", "/api/v1/alerts/{alert_id}/resolve"),
    ("GET", "/api/v1/monitoring/batches"), ("POST", "/api/v1/monitoring/sites/{site}/trigger"),
    ("GET", "/api/v1/monitoring/status"),
    ("GET", "/api/v1/analytics/price-trends"), ("GET", "/api/v1/analytics/volatile-products"),
    ("GET", "/api/v1/analytics/price-drops"), ("GET", "/api/v1/analytics/alerts-summary"),
    ("GET", "/api/v1/analytics/availability-timeline"), ("GET", "/api/v1/analytics/sites-overview"),
    ("POST", "/api/v1/search/text"), ("GET", "/api/v1/search/quick"),
    ("GET", "/api/v1/search/suggestions"), ("POST", "/api/v1/search/semantic"),
    ("GET", "/health"), ("GET", "/ready"), ("GET", "/metrics"),
}

WEBSOCKET_ROUTES = {"/api/v1/ws/progress", "/api/v1/ws/alerts"}


def test_every_public_route_is_registered_and_logged(caplog):
    app = create_app()
    registered_http = {
        (method, route.path)
        for route in app.routes if isinstance(route, APIRoute)
        for method in route.methods
    }
    registered_websockets = {route.path for route in app.routes if isinstance(route, APIWebSocketRoute)}

    missing_http = HTTP_ROUTES - registered_http
    missing_websockets = WEBSOCKET_ROUTES - registered_websockets
    for method, path in sorted(HTTP_ROUTES):
        logger.info("route_status method=%s path=%s status=%s", method, path, "registered" if (method, path) in registered_http else "missing")
    for path in sorted(WEBSOCKET_ROUTES):
        logger.info("route_status websocket path=%s status=%s", path, "registered" if path in registered_websockets else "missing")

    assert not missing_http
    assert not missing_websockets
