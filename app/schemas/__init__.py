"""Pydantic schemas package."""
from app.schemas.alert import (
    AlertResolveResponse,
    AlertResponse,
    PaginatedAlerts,
)
from app.schemas.analytics import (
    AlertSummary,
    AvailabilityTimelinePoint,
    AvailabilityTimelineResponse,
    PriceDrop,
    PriceDropsResponse,
    PriceTrendPoint,
    PriceTrendResponse,
    SiteOverview,
    SitesOverviewResponse,
    VolatileProduct,
    VolatileProductsResponse,
)
from app.schemas.auth import (
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)
from app.schemas.common import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    MessageResponse,
    PaginatedResponse,
    PaginationParams,
)
from app.schemas.product import (
    ChangeHistoryItem,
    ChangeHistoryResponse,
    PaginatedProducts,
    PriceHistoryItem,
    PriceHistoryResponse,
    ProductDetail,
    ProductSummary,
)
from app.schemas.search import (
    SearchResponse,
    SearchResult,
    SemanticSearchRequest,
    SuggestionItem,
    SuggestionsResponse,
    TextSearchRequest,
)

__all__ = [
    # auth
    "UserRegister", "UserLogin", "UserResponse", "TokenResponse",
    # product
    "ProductSummary", "ProductDetail", "PaginatedProducts",
    "PriceHistoryItem", "PriceHistoryResponse",
    "ChangeHistoryItem", "ChangeHistoryResponse",
    # alert
    "AlertResponse", "PaginatedAlerts", "AlertResolveResponse",
    # analytics
    "PriceTrendPoint", "PriceTrendResponse",
    "VolatileProduct", "VolatileProductsResponse",
    "AlertSummary",
    "PriceDrop", "PriceDropsResponse",
    "AvailabilityTimelinePoint", "AvailabilityTimelineResponse",
    "SiteOverview", "SitesOverviewResponse",
    # search
    "TextSearchRequest", "SemanticSearchRequest",
    "SearchResult", "SearchResponse",
    "SuggestionItem", "SuggestionsResponse",
    # common
    "PaginationParams", "PaginatedResponse",
    "ErrorDetail", "ErrorResponse",
    "MessageResponse", "HealthResponse",
]