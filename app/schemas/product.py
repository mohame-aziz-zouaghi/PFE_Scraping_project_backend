"""Product schemas — responses for /products endpoints."""
from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ═══════════════════════════════════════════════════════
# List / summary
# ═══════════════════════════════════════════════════════
class ProductSummary(BaseModel):
    """Compact product row for list views and cards."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    site: str
    main_product_id: Optional[str] = None
    special_id: Optional[str] = None

    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None

    price: Optional[float] = None
    discount_price: Optional[float] = None
    discount_amount: Optional[float] = None
    availability: Optional[str] = None

    url: str
    main_image: Optional[str] = None

    last_updated_at: Optional[datetime] = None
    last_checked: Optional[datetime] = None


# ═══════════════════════════════════════════════════════
# Detail — every column on `products`
# ═══════════════════════════════════════════════════════
class ProductDetail(BaseModel):
    """Full product row. Mirrors the `products` table."""

    model_config = ConfigDict(from_attributes=True)

    # Identity
    id: UUID
    site: str
    main_product_id: Optional[str] = None
    special_id: Optional[str] = None

    # Descriptive
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None

    # Pricing
    price: Optional[float] = None
    discount_price: Optional[float] = None
    discount_amount: Optional[float] = None
    availability: Optional[str] = None
    price_valid_until: Optional[date] = None

    # Content
    url: str
    description: Optional[str] = None
    short_description: Optional[Any] = None
    main_image: Optional[str] = None
    images: Optional[Any] = None

    # Lifecycle
    last_updated_at: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    is_enabled: Optional[bool] = None

    # Monitoring intelligence
    monitoring_priority: Optional[int] = None
    monitoring_group: Optional[str] = None
    next_check_at: Optional[datetime] = None
    check_failures: Optional[int] = None
    consecutive_errors: Optional[int] = None
    total_checks: Optional[int] = None
    total_errors: Optional[int] = None

    # Price history stats
    last_price_change: Optional[datetime] = None
    price_change_count: Optional[int] = None
    price_change_percent: Optional[float] = None
    price_volatility_score: Optional[int] = None
    avg_price: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    price_trend: Optional[str] = None
    change_frequency_7d: Optional[int] = None
    change_frequency_30d: Optional[int] = None
    days_since_last_change: Optional[int] = None

    # Availability history
    availability_change_count: Optional[int] = None
    last_availability_change: Optional[datetime] = None

    # Canonical link (Phase 3)
    canonical_id: Optional[UUID] = None


# ═══════════════════════════════════════════════════════
# Pagination wrapper
# ═══════════════════════════════════════════════════════
class PaginatedProducts(BaseModel):
    items: list[ProductSummary]
    total: int
    page: int
    page_size: int
    total_pages: int

    # Set when the response was truncated for anonymous visitors.
    # The frontend uses this to open the sign-in modal.
    requires_auth: bool = False
    auth_reason: Optional[str] = None


# ═══════════════════════════════════════════════════════
# History models (unchanged from your current file)
# ═══════════════════════════════════════════════════════
class PriceHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    price: float
    discount_price: Optional[float] = None
    discount_amount: Optional[float] = None
    availability: str
    recorded_at: datetime
    change_reason: Optional[str] = None


class PriceHistoryResponse(BaseModel):
    product_id: UUID
    items: list[PriceHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class ChangeHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    change_type: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    source: str
    change_severity: Optional[str] = None
    created_at: datetime
    monitored_at: Optional[datetime] = None


class ChangeHistoryResponse(BaseModel):
    product_id: UUID
    items: list[ChangeHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# ═══════════════════════════════════════════════════════
# Similar products (embedding KNN)
# ═══════════════════════════════════════════════════════
class SimilarProduct(BaseModel):
    product_id: UUID
    site: str
    main_product_id: Optional[str] = None
    special_id: Optional[str] = None
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    discount_price: Optional[float] = None
    availability: Optional[str] = None
    main_image: Optional[str] = None
    url: str
    similarity: float
    cosine_distance: float


class SimilarProductsResponse(BaseModel):
    query_product_id: UUID
    count: int
    items: list[SimilarProduct]


class EmbeddingStatsResponse(BaseModel):
    total_embedded: int
    total_pending: int
    status_breakdown: dict
    model: str