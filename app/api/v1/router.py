"""Main v1 router."""
from fastapi import APIRouter

from app.api.v1 import (
    alerts,
    analytics,
    auth,
    monitoring,
    products,
    search,
    websocket,
)
from app.core.honeypot import register_honeypot_routes

api_router = APIRouter()

# Real routes
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(products.router, prefix="/products", tags=["products"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(websocket.router, tags=["websocket"])

# Honeypot routes (must be LAST — they catch wild paths)
register_honeypot_routes(api_router)