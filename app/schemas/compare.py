# app/schemas/compare.py
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
    # NEW -- so the frontend can compute "is this the cheapest listing"
    effective_price: Optional[float] = None


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


class MatchStatsResponse(BaseModel):
    """
    Wrapper around whatever MatchingStats.as_dict() returns.

    All known fields have defaults so a partial stats dict (e.g. from
    a stats query that returns only persisted counters) still
    validates. `extra="allow"` passes anything else through.
    """
    model_config = ConfigDict(extra="allow")

    candidates_found: int = 0
    auto_accepted: int = 0
    rule_rejected: int = 0
    variant_rejected: int = 0
    llm_checked: int = 0
    llm_accepted: int = 0
    llm_rejected: int = 0
    llm_errors: int = 0
    llm_deferred: int = 0
    cluster_rejected: int = 0
    variant_groups_created: int = 0
    products_grouped: int = 0
    base_groups_created: int = 0
    base_groups_updated: int = 0
    pairs_linked: int = 0
    integrity_errors: int = 0
    errors: int = 0


class MatchRunStatusResponse(BaseModel):
    """Response for /compare/match/status."""
    active: bool
    progress: dict = {}