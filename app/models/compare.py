"""Schemas for /compare endpoints."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class GroupListing(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    product_id: UUID
    site: str
    main_product_id: Optional[str] = None
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    discount_price: Optional[float] = None
    discount_amount: Optional[float] = None
    availability: Optional[str] = None
    url: str
    main_image: Optional[str] = None
    last_checked: Optional[datetime] = None


class VariantInBase(BaseModel):
    group_id: UUID
    canonical_name: str
    variant_signature: Optional[dict] = None
    product_count: int
    site_count: int
    cheapest_price: Optional[float] = None
    cheapest_site: Optional[str] = None


class GroupDetail(BaseModel):
    group_id: UUID
    canonical_name: str
    canonical_brand: Optional[str] = None
    canonical_category: Optional[str] = None
    variant_signature: Optional[dict] = None
    product_count: int
    site_count: int
    confidence: float
    base_group_id: Optional[UUID] = None
    cheapest_site: Optional[str] = None
    cheapest_price: Optional[float] = None
    most_expensive_price: Optional[float] = None
    savings_amount: Optional[float] = None
    savings_percent: Optional[float] = None
    listings: list[GroupListing]


class BaseGroupDetail(BaseModel):
    base_group_id: UUID
    base_name: str
    base_brand: Optional[str] = None
    base_category: Optional[str] = None
    variant_count: int
    site_count: int
    total_product_count: int
    overall_cheapest_price: Optional[float] = None
    overall_cheapest_site: Optional[str] = None
    variants: list[VariantInBase]


# ── Base-level cheapest (Gap 5 from the audit) ─────────────
class CheapestResult(BaseModel):
    base_group_id: UUID
    base_name: str
    variant_count: int
    site_count: int
    cheapest_site: str
    cheapest_price: float


class CheapestResponse(BaseModel):
    query: str
    count: int
    results: list[CheapestResult]


class GroupHistoryPoint(BaseModel):
    date: str
    site: str
    avg_price: float
    min_price: float
    sample_count: int


class GroupHistoryResponse(BaseModel):
    group_id: UUID
    canonical_name: str
    period_days: int
    points: list[GroupHistoryPoint]


class ProductGroupResponse(BaseModel):
    product_id: UUID
    group_id: Optional[UUID] = None
    base_group_id: Optional[UUID] = None
    canonical_name: Optional[str] = None
    site_count: int = 0
    listings: list[GroupListing] = []


# ── Match stats — mirrors MatchingStats.as_dict() (Gap 8) ──
class MatchStatsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidates_found: int
    auto_accepted: int
    rule_rejected: int
    variant_rejected: int
    llm_checked: int
    llm_accepted: int
    llm_rejected: int
    llm_errors: int
    llm_deferred: int
    cluster_rejected: int
    variant_groups_created: int
    products_grouped: int
    base_groups_created: int
    base_groups_updated: int
    pairs_linked: int
    integrity_errors: int
    errors: int