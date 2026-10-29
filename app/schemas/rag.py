# app/schemas/rag.py
"""
Pydantic schemas for the RAG assistant.

Request and response shapes for /api/v1/rag/*.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ═══════════════════════════════════════════════════════
# Chat — request
# ═══════════════════════════════════════════════════════
class RagChatRequest(BaseModel):
    """
    One chat turn.

    `message = None` is the identity-layer call: the backend
    responds with the static welcome message without touching
    the LLM.

    `session_id` must be a UUID v4 string (validated by the
    endpoint, not here, so the schema stays usable in tests
    without a live endpoint).
    """

    session_id: str = Field(
        min_length=8,
        max_length=64,
        description="Client-generated session identifier. "
        "Not required to be a UUID, but must be stable across "
        "turns within the same browser session.",
    )
    message: Optional[str] = Field(
        default=None,
        max_length=500,
        description="The user's question, or None to request "
        "the welcome message.",
    )
    locale: Optional[str] = Field(
        default="en",
        max_length=8,
        description="Language hint. One of en, fr, ar. "
        "Anything else falls through as en.",
    )

    @field_validator("locale")
    @classmethod
    def _normalize_locale(cls, v: Optional[str]) -> str:
        if not v:
            return "en"
        v = v.strip().lower()
        if v not in ("en", "fr", "ar"):
            return "en"
        return v

    @field_validator("message")
    @classmethod
    def _strip_message(cls, v: Optional[str]) -> Optional[str]:
        # We do NOT validate length here beyond the Field
        # max_length. The InputGuardrail truncates to the
        # actual configured max (500 by default); Pydantic's
        # max_length just rejects absurd payloads early.
        if v is None:
            return None
        return v.strip()


# ═══════════════════════════════════════════════════════
# Chat — source cards
# ═══════════════════════════════════════════════════════
class RagSourceCard(BaseModel):
    """
    One card rendered under a bot answer.

    `kind` distinguishes product cards (a single listing) from
    canonical cards (a group of listings across sites). Fields
    not relevant to the kind are None.

    Kept as a single model rather than a discriminated union so
    cache rehydration (`RagSourceCard.model_validate(c)`) works
    without a `discriminator=` config.
    """

    kind: Literal["product", "canonical"]

    # ── Common ────────────────────────────────────────
    name: str

    # ── Product card ──────────────────────────────────
    product_id: Optional[UUID] = None
    site: Optional[str] = None
    price: Optional[float] = None
    availability: Optional[str] = None
    main_image: Optional[str] = None
    url: Optional[str] = None

    # ── Canonical card ────────────────────────────────
    canonical_id: Optional[UUID] = None
    site_count: Optional[int] = None
    cheapest_site: Optional[str] = None
    # `price` above is reused for canonical.cheapest_price.


# ═══════════════════════════════════════════════════════
# Chat — response
# ═══════════════════════════════════════════════════════
class RagChatResponse(BaseModel):
    """
    One chat turn's response.

    `remaining_anonymous` is:
      • an int for anonymous sessions (0..ANON_LIMIT)
      • None for authenticated sessions
    """

    answer: str
    source_cards: list[RagSourceCard] = Field(default_factory=list)

    cached: bool = False
    refused: bool = False
    is_welcome: bool = False
    requires_auth: bool = False

    remaining_anonymous: Optional[int] = None

    session_id: str
    latency_ms: int = 0
    request_id: str


# ═══════════════════════════════════════════════════════
# FAQ
# ═══════════════════════════════════════════════════════
class RagFaqItem(BaseModel):
    """One suggestion chip."""

    slug: str
    title: str
    category: str


class RagFaqResponse(BaseModel):
    """Response for GET /rag/faq."""

    locale: str
    count: int
    items: list[RagFaqItem] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════
# Feedback
# ═══════════════════════════════════════════════════════
class RagFeedbackRequest(BaseModel):
    """
    Thumbs up / down on a single turn.

    `turn_id` is the `request_id` returned by the chat response.
    """

    turn_id: str = Field(min_length=8, max_length=64)
    helpful: bool
    comment: Optional[str] = Field(default=None, max_length=500)

    @field_validator("comment")
    @classmethod
    def _strip_comment(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class RagFeedbackResponse(BaseModel):
    accepted: bool
    turn_id: str


# ═══════════════════════════════════════════════════════
# History
# ═══════════════════════════════════════════════════════
class RagHistoryItem(BaseModel):
    """One past RAG turn for the history endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    query: str
    hit: Optional[str] = None        # "cache" | "retrieval" | "refusal"
    latency_ms: Optional[int] = None
    weight: float = 3.0


class RagHistoryResponse(BaseModel):
    """Response for GET /rag/history."""

    user_id: Optional[UUID] = None
    count: int
    page: int = 1
    page_size: int = 20
    total_pages: int = 1
    items: list[RagHistoryItem] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════
# Internal — cached response envelope
# ═══════════════════════════════════════════════════════
class CachedRagAnswer(BaseModel):
    """
    The shape stored in Redis under the exact / semantic keys.

    Kept as a Pydantic model so the serializer and deserializer
    share one definition. The stored JSON is
    `CachedRagAnswer.model_dump(mode="json")`.
    """

    answer: str
    source_cards: list[RagSourceCard] = Field(default_factory=list)
    refused: bool = False
    is_welcome: bool = False
    cached_at: Optional[str] = None    # ISO timestamp, for debug


# ═══════════════════════════════════════════════════════
# Error envelope
# ═══════════════════════════════════════════════════════
class RagErrorResponse(BaseModel):
    """
    Shape used for 4xx/5xx from the RAG endpoints.

    Matches the app-wide error envelope used elsewhere
    (`{"error": {"code": ..., "message": ...}}`) so the
    frontend can handle it uniformly.
    """

    error: dict[str, Any]