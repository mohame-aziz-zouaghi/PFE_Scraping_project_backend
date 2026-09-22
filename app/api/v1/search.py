"""Search endpoints — text + suggestions (semantic ready for future)."""
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user_optional
from app.core.rate_limit import rate_limit_anon, rate_limit_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.search import (
    SearchResponse,
    SemanticSearchRequest,
    SuggestionsResponse,
    TextSearchRequest,
)
from app.services.search_service import SearchService

router = APIRouter()


@router.post("/text", response_model=SearchResponse,
                 dependencies=[rate_limit_anon(times=60, seconds=60)])
async def text_search(
    request: Request,
    payload: TextSearchRequest,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
):
    """Full-text search over products."""
    return await SearchService(db).text_search(payload)


@router.get("/quick", response_model=SearchResponse,
                dependencies=[rate_limit_anon(times=20, seconds=60)])
async def quick_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200),
    site: Optional[str] = Query(None, max_length=50),
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
):
    """Quick GET-based search (for autocomplete UIs)."""
    payload = TextSearchRequest(
        query=q, site=site, page=1, page_size=limit, sort_by="relevance"
    )
    return await SearchService(db).text_search(payload)


@router.get("/suggestions", response_model=SuggestionsResponse,
                dependencies=[rate_limit_anon(times=200, seconds=60)])
async def suggestions(
    request: Request,
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(8, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
):
    """Autocomplete suggestions for product/brand names."""
    return await SearchService(db).suggestions(query=q, limit=limit)


@router.post("/semantic", response_model=SearchResponse,
                 dependencies=[rate_limit_anon(times=30, seconds=60)])
async def semantic_search(
    request: Request,
    payload: SemanticSearchRequest,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Semantic search using pgvector embeddings.
    Falls back to text search if embeddings aren't available.
    """
    return await SearchService(db).semantic_search(payload)