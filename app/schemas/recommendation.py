# app/schemas/recommendation.py
"""Recommendation schemas — event logging + responses."""
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Fixed set of event types. Must match the SQL CHECK constraint.
EventType = Literal[
    "view", "click", "dwell", "wishlist", "add_to_cart", "search"
]

# Allowed keys inside `metadata`, per event type. Anything not in
# these lists is rejected — this is what keeps PII out of the DB.
_METADATA_KEYS: dict[str, set[str]] = {
    "view": {"referrer", "position"},
    "click": {"position", "source"},
    "dwell": {"seconds"},
    "wishlist": set(),
    "add_to_cart": {"qty"},
    "search": {"query", "results_count"},
}


class EventIn(BaseModel):
    """One event inside a batch."""

    type: EventType
    product_id: Optional[UUID] = None
    ts: datetime = Field(
        default_factory=lambda: __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        description="Client-side timestamp. Server uses created_at "
        "regardless, but keeps this for ordering analysis.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(
        cls, v: dict[str, Any], info
    ) -> dict[str, Any]:
        event_type = info.data.get("type")
        allowed = _METADATA_KEYS.get(event_type or "", set())
        for key in v:
            if key not in allowed:
                raise ValueError(
                    f"metadata key '{key}' not allowed for "
                    f"event type '{event_type}'"
                )
        if event_type == "dwell":
            seconds = v.get("seconds")
            if seconds is not None:
                if not isinstance(seconds, (int, float)):
                    raise ValueError("dwell.seconds must be numeric")
                if seconds < 0 or seconds > 3600:
                    raise ValueError(
                        "dwell.seconds must be between 0 and 3600"
                    )
        if event_type == "add_to_cart":
            qty = v.get("qty")
            if qty is not None:
                if not isinstance(qty, int):
                    raise ValueError("add_to_cart.qty must be an int")
                if qty < 1 or qty > 100:
                    raise ValueError(
                        "add_to_cart.qty must be between 1 and 100"
                    )
        if event_type == "search":
            q = v.get("query")
            if not q or not isinstance(q, str):
                raise ValueError(
                    "search.query is required and must be a string"
                )
            q = q.strip().lower()
            if not (1 <= len(q) <= 120):
                raise ValueError("search.query must be 1–120 chars")
            v["query"] = q
        return v

    @model_validator(mode="after")
    def _require_product_unless_search(self):
        if self.type == "search":
            if not (self.metadata or {}).get("query"):
                raise ValueError(
                    "search event requires metadata.query"
                )
            return self
        if self.product_id is None:
            raise ValueError(
                f"event type '{self.type}' requires product_id"
            )
        return self


class EventsBatchIn(BaseModel):
    """Payload for POST /recommendations/events."""

    session_id: Optional[str] = Field(default=None, max_length=64)
    events: list[EventIn] = Field(min_length=1, max_length=50)

    @field_validator("session_id")
    @classmethod
    def _clean_session(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v[:64] or None


class EventsBatchOut(BaseModel):
    accepted: int
    rejected: int
    errors: list[dict[str, Any]] = Field(default_factory=list)


class EventOut(BaseModel):
    """Internal representation, useful for future endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: UUID
    event_type: str
    product_id: Optional[UUID] = None
    canonical_id: Optional[UUID] = None
    base_group_id: Optional[UUID] = None
    metadata: dict[str, Any] = Field(
        default_factory=dict, alias="metadata_"
    )
    weight: float
    session_id: Optional[str] = None
    created_at: datetime


# ═══════════════════════════════════════════════════════
# Recommendations — response shapes
# ═══════════════════════════════════════════════════════
class RecommendedItem(BaseModel):
    product_id: UUID
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None

    price: Optional[float] = None
    discount_price: Optional[float] = None
    availability: Optional[str] = None

    main_image: Optional[str] = None
    url: str
    site: str

    rank: int
    score: float
    source: str
    reason: Optional[str] = None


class RecommendationsResponse(BaseModel):
    user_id: Optional[UUID] = None
    cold_start: bool = False
    cached: bool = False
    count: int
    items: list[RecommendedItem]


class SimilarProductsResponse(BaseModel):
    query_product_id: UUID
    count: int
    items: list[RecommendedItem]