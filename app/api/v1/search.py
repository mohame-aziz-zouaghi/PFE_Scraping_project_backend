"""Search endpoints — text, suggestions, semantic."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user_optional
from app.core.cache import Cache
from app.core.logging_config import get_logger
from app.core.rate_limit import rate_limit_anon
from app.db.session import get_db
from app.models.user import User
from app.schemas.search import (
    SearchResponse,
    SemanticSearchRequest,
    SuggestionsResponse,
    TextSearchRequest,
)
from app.services.search_service import (
    SearchService,
    semantic_search_cache_key,
    suggestions_cache_key,
    text_search_cache_key,
)

logger = get_logger(__name__)
router = APIRouter()

TTL_TEXT_SEARCH = 300
TTL_SUGGESTIONS = 120
TTL_SEMANTIC_SEARCH = 900


@router.post(
    "/text",
    response_model=SearchResponse,
    dependencies=[rate_limit_anon(times=60, seconds=60)],
)
async def text_search(
    request: Request,
    payload: TextSearchRequest,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> SearchResponse:
    # Logged-in users bypass the cache so search events fire cleanly.
    if user is not None:
        return await SearchService(db).text_search(payload, user_id=user.id)

    key = text_search_cache_key(payload)
    cached = await Cache.get(key)
    if cached is not None:
        return SearchResponse.model_validate(cached)

    result = await SearchService(db).text_search(payload)
    await Cache.set(key, result.model_dump(mode="json"), TTL_TEXT_SEARCH)
    return result


@router.get(
    "/quick",
    response_model=SearchResponse,
    dependencies=[rate_limit_anon(times=20, seconds=60)],
)
async def quick_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200),
    site: Optional[str] = Query(None, max_length=50),
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> SearchResponse:
    payload = TextSearchRequest(
        query=q, site=site, page=1, page_size=limit, sort_by="relevance",
    )
    if user is not None:
        return await SearchService(db).text_search(payload, user_id=user.id)

    key = text_search_cache_key(payload)
    cached = await Cache.get(key)
    if cached is not None:
        return SearchResponse.model_validate(cached)

    result = await SearchService(db).text_search(payload)
    await Cache.set(key, result.model_dump(mode="json"), TTL_TEXT_SEARCH)
    return result


@router.get(
    "/suggestions",
    response_model=SuggestionsResponse,
    dependencies=[rate_limit_anon(times=200, seconds=60)],
)
async def suggestions(
    request: Request,
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(8, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
) -> SuggestionsResponse:
    key = suggestions_cache_key(q, limit)
    cached = await Cache.get(key)
    if cached is not None:
        return SuggestionsResponse.model_validate(cached)

    result = await SearchService(db).suggestions(query=q, limit=limit)
    await Cache.set(key, result.model_dump(mode="json"), TTL_SUGGESTIONS)
    return result


@router.post(
    "/semantic",
    response_model=SearchResponse,
    dependencies=[rate_limit_anon(times=30, seconds=60)],
)
async def semantic_search(
    request: Request,
    payload: SemanticSearchRequest,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> SearchResponse:
    if user is not None:
        return await SearchService(db).semantic_search(payload, user_id=user.id)

    key = semantic_search_cache_key(payload)
    cached = await Cache.get(key)
    if cached is not None:
        return SearchResponse.model_validate(cached)

    result = await SearchService(db).semantic_search(payload)
    await Cache.set(key, result.model_dump(mode="json"), TTL_SEMANTIC_SEARCH)
    return result