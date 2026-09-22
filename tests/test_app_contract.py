from app.main import create_app


def test_health_and_readiness_routes_are_registered():
    paths = {route.path for route in create_app().routes}
    assert "/health" in paths
    assert "/ready" in paths
    assert "/api/v1/monitoring/status" in paths
