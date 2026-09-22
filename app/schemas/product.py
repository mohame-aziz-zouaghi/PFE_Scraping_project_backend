"""Product schemas."""
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID as _UUID
from typing import Optional as _Optional
from pydantic import BaseModel, ConfigDict


class ProductSummary(BaseModel):
    """Compact product representation for lists."""
    model_config = ConfigDict(from_attributes=True)

    id: _UUID
    site: str
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    price: float
    discount_price: Optional[float] = None
    discount_amount: Optional[float] = None
    availability: str
    main_image: Optional[str] = None
    last_updated_at: datetime


class ProductDetail(ProductSummary):
    """Full product representation."""
    main_product_id: str
    url: str
    description: Optional[str] = None
    short_description: Optional[Any] = None
    images: Optional[List[str]] = None
    price_valid_until: Optional[date] = None
    is_enabled: bool
    last_checked: Optional[datetime] = None


class PaginatedProducts(BaseModel):
    items: List[ProductSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class PriceHistoryItem(BaseModel):
    """Single price history record."""
    model_config = ConfigDict(from_attributes=True)

    id: _UUID
    price: float
    discount_price: Optional[float] = None
    discount_amount: Optional[float] = None
    availability: str
    recorded_at: datetime
    change_reason: Optional[str] = None


class PriceHistoryResponse(BaseModel):
    product_id: _UUID
    items: List[PriceHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class ChangeHistoryItem(BaseModel):
    """Single product change event."""
    model_config = ConfigDict(from_attributes=True)

    id: _UUID
    change_type: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    change_severity: Optional[str] = None
    source: Optional[str] = None
    created_at: datetime


class ChangeHistoryResponse(BaseModel):
    product_id: _UUID
    items: List[ChangeHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int




class SimilarProduct(BaseModel):
    """A product returned by the /similar endpoint."""

    product_id: _UUID
    site: str
    main_product_id: str
    name: str
    brand: _Optional[str] = None
    category: _Optional[str] = None
    price: _Optional[float] = None
    discount_price: _Optional[float] = None
    availability: _Optional[str] = None
    main_image: _Optional[str] = None
    url: _Optional[str] = None
    similarity: float
    cosine_distance: float

    model_config = ConfigDict(from_attributes=True)


class SimilarProductsResponse(BaseModel):
    """Response shape for GET /products/{id}/similar."""

    query_product_id: _UUID
    count: int
    items: list[SimilarProduct]


class EmbeddingStatsResponse(BaseModel):
    """Diagnostic endpoint output."""

    total_embedded: int
    total_pending: int
    status_breakdown: dict[str, int]
    model: str
