"""Analytics schemas."""
from datetime import date, datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class PriceTrendPoint(BaseModel):
    date: date
    avg_price: float
    min_price: float
    max_price: float
    count: int


class PriceTrendResponse(BaseModel):
    product_id: Optional[UUID] = None
    site: Optional[str] = None
    period_days: int
    points: List[PriceTrendPoint]


class VolatileProduct(BaseModel):
    product_id: UUID
    site: str
    name: str
    url: str
    price: float
    price_volatility_score: int
    price_change_count: int
    last_price_change: Optional[datetime] = None


class VolatileProductsResponse(BaseModel):
    period_days: int
    items: List[VolatileProduct]
    total: int


class AlertSummary(BaseModel):
    """Summary counts by site/type/severity."""
    total: int
    unresolved: int
    resolved: int
    by_site: Dict[str, int]
    by_type: Dict[str, int]
    by_severity: Dict[str, int]


class PriceDrop(BaseModel):
    product_id: UUID
    site: str
    name: str
    url: str
    old_price: float
    new_price: float
    change_percent: float
    detected_at: datetime


class PriceDropsResponse(BaseModel):
    period_days: int
    items: List[PriceDrop]
    total: int


class AvailabilityTimelinePoint(BaseModel):
    date: date
    in_stock: int
    out_of_stock: int
    on_order: int


class AvailabilityTimelineResponse(BaseModel):
    site: Optional[str] = None
    period_days: int
    points: List[AvailabilityTimelinePoint]


class SiteOverview(BaseModel):
    site: str
    total_products: int
    enabled_products: int
    in_stock: int
    out_of_stock: int
    avg_price: float
    last_batch_status: Optional[str] = None
    last_batch_at: Optional[datetime] = None
    unresolved_alerts: int


class SitesOverviewResponse(BaseModel):
    items: List[SiteOverview]
    total_sites: int