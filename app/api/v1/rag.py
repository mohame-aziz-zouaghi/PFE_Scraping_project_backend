# app/api/v1/rag.py
"""
RAG assistant endpoints.

Base path: /api/v1/rag

    POST   /chat       — one chat turn (optional auth)
    GET    /faq        — top suggestion chips
    GET    /history    — paginated user history (auth required)
    DELETE /history    — clear the user's RAG history (auth required)
    POST   /feedback   — thumbs up/down (auth required)

Rate limits:
    • /chat          — anon: 30/min, auth: 60/min (mirrors semantic_search)
    • /faq           — 60/min per IP
    • /history       — 60/min per user
    • /feedback      — 20/min per user

All responses use the app-wide Pydantic models or the standard
error envelope (`{"error": {"code": ..., "message": ..., "details": {...}}}`).
"""
from __future__ import annotations

import re
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.deps import get_current_user, get_current_user_optional
from app.core.logging_config import get_logger
from app.core.rate_limit import rate_limit_anon, rate_limit_user
from app.core.rag_config import RAG_ANON
from app.models.help_article import HelpArticle
from app.models.user import User
from app.models.user_events import UserEvent
from app.schemas.rag import (
    RagChatRequest,
    RagChatResponse,
    RagFaqItem,
    RagFaqResponse,
    RagFeedbackRequest,
    RagFeedbackResponse,
    RagHistoryItem,
    RagHistoryResponse,
)
from app.services.rag_service import RAGService

logger = get_logger(__name__)

router = APIRouter()


# ═══════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════
_UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _validate_session_id(session_id: str) -> str:
    """
    Enforce UUID v4. The design doc §11 requires this so that
    the anon counter can't be keyed by an attacker-controlled
    arbitrary string.

    Accepts any well-formed UUID v4 string. Rejects anything
    else with a 400.
    """
    sid = (session_id or "").strip()
    if not _UUID_V4_RE.match(sid):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_session_id",
                    "message": (
                        "session_id must be a UUID v4 string."
                    ),
                    "details": {},
                }
            },
        )
    return sid


# ═══════════════════════════════════════════════════════
# POST /rag/chat
# ═══════════════════════════════════════════════════════
@router.post(
    "/chat",
    response_model=RagChatResponse,
    dependencies=[rate_limit_anon(times=30, seconds=60)],
)
async def rag_chat(
    payload: RagChatRequest,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> RagChatResponse:
    """
    One chat turn.

    • `payload.message = null` → welcome message (no LLM).
    • `payload.message = "..."` → full pipeline.

    Authenticated users bypass the anon counter. Anonymous
    sessions consume a slot per non-welcome, non-refused turn.
    """
    session_id = _validate_session_id(payload.session_id)

    service = RAGService(db)
    try:
        return await service.chat(
            request=payload.model_copy(
                update={"session_id": session_id}
            ),
            user_id=user.id if user else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # The service is designed never to raise. This branch
        # catches bugs so the client gets a clean 500 with a
        # request_id rather than a stack trace.
        logger.exception(
            "rag_chat_unhandled",
            session_id=session_id,
            error=str(exc)[:200],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                    "details": {},
                }
            },
        )


# ═══════════════════════════════════════════════════════
# GET /rag/faq
# ═══════════════════════════════════════════════════════
@router.get(
    "/faq",
    response_model=RagFaqResponse,
    dependencies=[rate_limit_anon(times=60, seconds=60)],
)
async def rag_faq(
    locale: str = Query(default="en", max_length=8),
    limit: int = Query(default=10, ge=1, le=30),
    db: AsyncSession = Depends(get_db),
) -> RagFaqResponse:
    """
    Return the top suggestion chips for the welcome message.

    Only `general`-category help articles are returned — those
    are the ones safe to surface to anonymous users.
    """
    locale_norm = (locale or "en").strip().lower()
    if locale_norm not in ("en", "fr", "ar"):
        locale_norm = "en"

    stmt = (
        select(
            HelpArticle.slug,
            HelpArticle.title,
            HelpArticle.category,
        )
        .where(
            HelpArticle.locale == locale_norm,
            HelpArticle.category == "general",
            HelpArticle.is_published.is_(True),
        )
        .order_by(HelpArticle.created_at.asc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()

    items = [
        RagFaqItem(
            slug=r.slug,
            title=r.title,
            category=r.category,
        )
        for r in rows
    ]
    return RagFaqResponse(
        locale=locale_norm,
        count=len(items),
        items=items,
    )


# ═══════════════════════════════════════════════════════
# GET /rag/history
# ═══════════════════════════════════════════════════════
@router.get(
    "/history",
    response_model=RagHistoryResponse,
    dependencies=[rate_limit_user(times=60, seconds=60)],
)
async def rag_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RagHistoryResponse:
    """
    Paginated history of the user's past RAG turns.

    Reads from `user_events` where `event_type = 'rag_query'`.
    Newest first.
    """
    offset = (page - 1) * page_size

    total = int(
        (
            await db.execute(
                select(func.count())
                .select_from(UserEvent)
                .where(
                    UserEvent.user_id == user.id,
                    UserEvent.event_type == "rag_query",
                )
            )
        ).scalar_one()
    )

    stmt = (
        select(UserEvent)
        .where(
            UserEvent.user_id == user.id,
            UserEvent.event_type == "rag_query",
        )
        .order_by(UserEvent.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    events = list((await db.execute(stmt)).scalars().all())

    items: list[RagHistoryItem] = []
    for e in events:
        meta = e.metadata_ or {}
        items.append(RagHistoryItem(
            id=e.id,
            created_at=e.created_at,
            query=meta.get("query", ""),
            hit=meta.get("hit"),
            latency_ms=meta.get("latency_ms"),
            weight=float(e.weight or 3.0),
        ))

    total_pages = max(1, (total + page_size - 1) // page_size)

    return RagHistoryResponse(
        user_id=user.id,
        count=len(items),
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        items=items,
    )


# ═══════════════════════════════════════════════════════
# DELETE /rag/history
# ═══════════════════════════════════════════════════════
@router.delete(
    "/history",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[rate_limit_user(times=10, seconds=60)],
)
async def rag_history_delete(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    await db.execute(
        delete(UserEvent).where(
            UserEvent.user_id == user.id,
            UserEvent.event_type.in_(("rag_query", "rag_feedback")),
        )
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ═══════════════════════════════════════════════════════
# POST /rag/feedback
# ═══════════════════════════════════════════════════════
@router.post(
    "/feedback",
    response_model=RagFeedbackResponse,
    dependencies=[rate_limit_user(times=20, seconds=60)],
)
async def rag_feedback(
    payload: RagFeedbackRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RagFeedbackResponse:
    """
    Record a thumbs up/down on a single turn.

    Writes one `rag_feedback` event with the turn_id and helpful
    flag in metadata. Used for offline quality review.
    """
    event = UserEvent(
        user_id=user.id,
        event_type="rag_feedback",
        product_id=None,
        metadata_={
            "turn_id": payload.turn_id,
            "helpful": payload.helpful,
            "comment": payload.comment,
        },
        weight=1.0,
    )
    db.add(event)
    await db.commit()

    return RagFeedbackResponse(
        accepted=True,
        turn_id=payload.turn_id,
    )