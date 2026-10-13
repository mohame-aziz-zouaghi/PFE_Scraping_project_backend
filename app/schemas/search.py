"""Search schemas."""
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Valid sort values for text search
SORT_BY_CHOICES = ("relevance", "price", "price_desc", "recent")


class TextSearchRequest(BaseModel):
    """Payload for POST /search/text."""
    query: str = Field(min_length=1, max_length=200)
    site: Optional[str] = Field(default=None, max_length=50)
    brand: Optional[str] = Field(default=None, max_length=200)
    category: Optional[str] = Field(default=None, max_length=200)
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    availability: Optional[str] = Field(default=None, max_length=20)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: str = Field(
        default="relevance",
        pattern="^(relevance|price|price_desc|recent)$",
    )
    session_id: Optional[str] = Field(default=None, max_length=64)



class SemanticSearchRequest(BaseModel):
    """Payload for POST /search/semantic."""
    query: str = Field(min_length=1, max_length=500)
    site: Optional[str] = Field(default=None, max_length=50)
    limit: int = Field(default=20, ge=1, le=100)
    min_similarity: float = Field(default=0.5, ge=0, le=1)
    session_id: Optional[str] = Field(default=None, max_length=64)



class SearchResult(BaseModel):
    """Single result item."""
    product_id: UUID
    site: str
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    price: float
    availability: str
    main_image: Optional[str] = None
    url: str
    score: float = Field(description="Relevance or similarity score")


class SearchResponse(BaseModel):
    """Response for any search endpoint."""
    query: str
    items: List[SearchResult]
    total: int
    page: int = 1
    page_size: int = 20
    total_pages: int = 1
    took_ms: float = 0.0


class SuggestionItem(BaseModel):
    """Autocomplete suggestion item."""
    text: str
    type: str = Field(description="product | brand | category")
    count: Optional[int] = None


class SuggestionsResponse(BaseModel):
    """Autocomplete suggestions response."""
    query: str
    items: List[SuggestionItem]