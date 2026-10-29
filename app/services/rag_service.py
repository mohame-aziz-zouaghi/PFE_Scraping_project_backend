# app/services/rag_service.py
"""
RAG assistant — top-level orchestrator.

Pipeline:

    1. Input guardrail
    2. Identity layer (welcome message)
    3. Cache lookup (exact + semantic, anon suggestion)
    4. Intent shortcut
    5. Direct help-article match (FAQ questions — no LLM)
    6. Standard retrieval (products + canonical + help)
    7. Relevance gate
    8. Did-you-mean (partial matches)
    9. Fallback generation (on-domain only)
   10. Refusal (off-domain)
   11. Signal log

Two public methods:

    chat(...)             — hot path, called from /api/v1/rag/chat
    answer_question(...)  — used by prewarm_anon_suggestions()
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.core.rag_config import (
    RAG,
    RAG_ANON,
    RAG_BUDGET_EXCEEDED_MESSAGE,
    RAG_UNAVAILABLE_MESSAGE,
    RAG_WELCOME_MESSAGE,
    build_system_prompt,
    refusal_for_category,
)
from app.models.help_article import HelpArticle
from app.models.user_events import UserEvent
from app.schemas.rag import (
    RagChatRequest,
    RagChatResponse,
    RagSourceCard,
)
from app.services.rag_cache import RagCache, normalize_message, rag_cache
from app.services.rag_guardrails import (
    InputGuardrail,
    OutputGuardrail,
)
from app.services.rag_intents import is_off_domain, is_on_domain
from app.services.rag_retrieval import (
    RetrievalService,
    RetrievedChunk,
)
from app.services.structured_queries import StructuredQueryService

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Concurrency semaphore
# ═══════════════════════════════════════════════════════
_llm_sem: Optional[asyncio.Semaphore] = None


def _get_sem() -> asyncio.Semaphore:
    global _llm_sem
    if _llm_sem is None:
        _llm_sem = asyncio.Semaphore(RAG.CONCURRENCY)
    return _llm_sem


# ═══════════════════════════════════════════════════════
# Internal types
# ═══════════════════════════════════════════════════════
@dataclass
class _PipelineResult:
    answer: str
    source_cards: list[RagSourceCard]
    refused: bool
    is_welcome: bool
    cached: bool
    retrieval_top_score: float
    chunk_ids: list[str]
    cache_source: Optional[str] = None


# ═══════════════════════════════════════════════════════
# RAGService
# ═══════════════════════════════════════════════════════
class RAGService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.cache: RagCache = rag_cache
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()
        self.retrieval = RetrievalService(db)
        self.structured = StructuredQueryService(db)      # ← new
        self.ollama = _OllamaClient(db=db)

    # ═══════════════════════════════════════════════════
    # Public — hot path
    # ═══════════════════════════════════════════════════
    async def chat(
        self,
        *,
        request: RagChatRequest,
        user_id: Optional[UUID] = None,
    ) -> RagChatResponse:
        start = time.perf_counter()
        session_id = request.session_id
        locale = (request.locale or "en").lower()

        # Welcome
        if request.message is None:
            return await self._welcome(
                session_id=session_id,
                user_id=user_id,
                locale=locale,
                start=start,
            )

        # Input guardrail
        verdict = self.input_guard.check(
            message=request.message,
            locale=locale,
            session_id=session_id,
        )
        if not verdict.allowed:
            remaining = await self._remaining_anonymous(
                session_id=session_id, user_id=user_id
            )
            return RagChatResponse(
                answer=refusal_for_category(None),
                source_cards=[],
                cached=False,
                refused=True,
                is_welcome=False,
                requires_auth=False,
                remaining_anonymous=remaining,
                session_id=session_id,
                latency_ms=self._ms_since(start),
                request_id=_new_request_id(),
            )

        # Anon cap
        if user_id is None:
            remaining = await self._remaining_anonymous(
                session_id=session_id, user_id=None
            )
            if remaining <= 0:
                return RagChatResponse(
                    answer=(
                        "You've reached the free limit. Sign in to "
                        "keep chatting — it's free."
                    ),
                    source_cards=[],
                    cached=False,
                    refused=False,
                    is_welcome=False,
                    requires_auth=True,
                    remaining_anonymous=0,
                    session_id=session_id,
                    latency_ms=self._ms_since(start),
                    request_id=_new_request_id(),
                )

        try:
            result = await self._run_pipeline(
                message=verdict.cleaned,
                locale=locale,
                user_id=user_id,
                session_id=session_id,
                is_anon_suggestion=False,
            )
        except Exception as exc:
            logger.exception(
                "rag_pipeline_unhandled", error=str(exc)[:200]
            )
            return RagChatResponse(
                answer=RAG_UNAVAILABLE_MESSAGE,
                source_cards=[],
                cached=False,
                refused=True,
                is_welcome=False,
                requires_auth=False,
                remaining_anonymous=await self._remaining_anonymous(
                    session_id=session_id, user_id=user_id
                ),
                session_id=session_id,
                latency_ms=self._ms_since(start),
                request_id=_new_request_id(),
            )

        remaining: Optional[int] = None
        if user_id is None and not result.refused:
            remaining = await self._consume_anon_slot(
                session_id=session_id
            )
        elif user_id is None:
            remaining = await self._remaining_anonymous(
                session_id=session_id, user_id=None
            )

        try:
            await self._log_turn(
                user_id=user_id,
                session_id=session_id,
                message=verdict.cleaned,
                result=result,
                latency_ms=self._ms_since(start),
            )
        except Exception as exc:
            logger.warning(
                "rag_signal_log_failed", error=str(exc)[:200]
            )

        return RagChatResponse(
            answer=result.answer,
            source_cards=result.source_cards,
            cached=result.cached,
            refused=result.refused,
            is_welcome=False,
            requires_auth=False,
            remaining_anonymous=remaining,
            session_id=session_id,
            latency_ms=self._ms_since(start),
            request_id=_new_request_id(),
        )

    # ═══════════════════════════════════════════════════
    # Public — prewarm
    # ═══════════════════════════════════════════════════
    async def answer_question(
        self,
        *,
        message: str,
        locale: str = "en",
        is_anon_suggestion: bool = False,
    ) -> RagChatResponse:
        start = time.perf_counter()
        verdict = self.input_guard.check(message=message, locale=locale)
        if not verdict.allowed:
            return RagChatResponse(
                answer=refusal_for_category(None),
                source_cards=[],
                cached=False,
                refused=True,
                is_welcome=False,
                requires_auth=False,
                remaining_anonymous=None,
                session_id="prewarm",
                latency_ms=self._ms_since(start),
                request_id=_new_request_id(),
            )

        result = await self._run_pipeline(
            message=verdict.cleaned,
            locale=locale,
            user_id=None,
            session_id="prewarm",
            is_anon_suggestion=is_anon_suggestion,
            skip_anon_cache=is_anon_suggestion,
        )
        return RagChatResponse(
            answer=result.answer,
            source_cards=result.source_cards,
            cached=result.cached,
            refused=result.refused,
            is_welcome=False,
            requires_auth=False,
            remaining_anonymous=None,
            session_id="prewarm",
            latency_ms=self._ms_since(start),
            request_id=_new_request_id(),
        )

    # ═══════════════════════════════════════════════════
    # Public — FAQ list
    # ═══════════════════════════════════════════════════
    async def list_anon_suggestions(
        self, *, locale: str = "en", limit: int = 10
    ) -> list[str]:
        stmt = (
            select(HelpArticle.title)
            .where(
                HelpArticle.locale == locale,
                HelpArticle.category == "general",
                HelpArticle.is_published.is_(True),
            )
            .order_by(HelpArticle.created_at.asc())
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).all()
        return [r[0] for r in rows]

    # ═══════════════════════════════════════════════════
    # The pipeline
    # ═══════════════════════════════════════════════════
    async def _run_pipeline(
        self,
        *,
        message: str,
        locale: str,
        user_id: Optional[UUID],
        session_id: str,
        is_anon_suggestion: bool = False,
        skip_anon_cache: bool = False,
    ) -> _PipelineResult:
        normalized = normalize_message(message)

        # ═══════════════════════════════════════════════════════
        # LAYER 4 — Cache
        # ═══════════════════════════════════════════════════════
        query_embedding: Optional[list[float]] = None
        cache_source: Optional[str] = None

        if user_id is None and not skip_anon_cache:
            hit = await self._safe_cache_lookup(
                message=message,
                locale=locale,
                query_embedding=None,
                anon_suggestion_only=True,
            )
            if hit is not None:
                cache_source = "anon"

        if cache_source is None:
            query_embedding = await self._embed(message)
            hit = await self._safe_cache_lookup(
                message=message,
                locale=locale,
                query_embedding=query_embedding,
                anon_suggestion_only=False,
            )
            if hit is not None:
                cache_source = "cache"

        if cache_source is not None:
            return self._pipeline_from_cache(
                payload=hit, cache_source=cache_source,
            )

        # ═══════════════════════════════════════════════════════
        # LAYER 3 — Intent shortcut
        # ═══════════════════════════════════════════════════════
        from app.services.rag_intents import match_intent

        intent = match_intent(message)
        if intent is not None:
            result = _PipelineResult(
                answer=intent.answer,
                source_cards=[],
                refused=False,
                is_welcome=False,
                cached=False,
                retrieval_top_score=1.0,
                chunk_ids=[f"intent:{intent.name}"],
            )
            if query_embedding is not None:
                await self._safe_cache_store(
                    message=message,
                    locale=locale,
                    payload=result,
                    query_embedding=query_embedding,
                )
            return result

        # ═══════════════════════════════════════════════════════
        # LAYER 3.25 — Structured query (aggregations)
        #
        # "cheapest PC", "most expensive laptop", "how many
        # phones", "laptops under 1300 DT" — these are SQL
        # queries, not semantic lookups. Answer directly from
        # the products table. Zero LLM, exact answer.
        # ═══════════════════════════════════════════════════════
        try:
            structured = await self.structured.try_handle(
                message=message, locale=locale,
            )
        except Exception as exc:
            logger.warning(
                "structured_query_layer_failed",
                error=str(exc)[:200],
            )
            structured = None

        if structured is not None and structured.handled:
            cards = [
                RagSourceCard.model_validate(c)
                for c in structured.source_cards
            ]
            result = _PipelineResult(
                answer=structured.answer,
                source_cards=cards,
                refused=False,
                is_welcome=False,
                cached=False,
                retrieval_top_score=1.0,
                chunk_ids=[f"structured:{structured.intent}"],
                cache_source="structured",
            )
            if query_embedding is not None:
                await self._safe_cache_store(
                    message=message,
                    locale=locale,
                    payload=result,
                    query_embedding=query_embedding,
                )
            return result

        # ═══════════════════════════════════════════════════════
        # LAYER 3.5 — Direct help-article match (FAQ)
        #
        # FAQ questions like "what is this site", "how do I
        # search", "which stores" have known written answers in
        # help_articles. If the query embedding is a very strong
        # match, return the article's body directly. Skip
        # retrieval, skip the LLM, skip the output guardrail.
        # Response time: <100ms.
        # ═══════════════════════════════════════════════════════
        if query_embedding is not None:
            help_match = await self.retrieval.find_help_article(
                query_embedding=query_embedding,
                locale=locale,
                min_similarity=0.72,
            )
            if help_match is not None:
                result = _PipelineResult(
                    answer=help_match.body.strip(),
                    source_cards=[],
                    refused=False,
                    is_welcome=False,
                    cached=False,
                    retrieval_top_score=help_match.cosine,
                    chunk_ids=[f"help:{help_match.slug}"],
                    cache_source="help_direct",
                )
                await self._safe_cache_store(
                    message=message,
                    locale=locale,
                    payload=result,
                    query_embedding=query_embedding,
                )
                return result

        # ═══════════════════════════════════════════════════════
        # LAYER 1 — Retrieval
        # ═══════════════════════════════════════════════════════
        if query_embedding is None:
            query_embedding = await self._embed(message)
        if query_embedding is None:
            return _PipelineResult(
                answer=RAG_UNAVAILABLE_MESSAGE,
                source_cards=[],
                refused=True,
                is_welcome=False,
                cached=False,
                retrieval_top_score=0.0,
                chunk_ids=[],
            )

        retrieval = await self.retrieval.retrieve(
            query_embedding=query_embedding,
            locale=locale,
        )

        # ═══════════════════════════════════════════════════════
        # Gate on the top score
        # ═══════════════════════════════════════════════════════
        if retrieval.top_score >= RAG.MIN_SIMILARITY:
            return await self._answer_from_retrieval(
                message=message,
                locale=locale,
                retrieval=retrieval,
                query_embedding=query_embedding,
            )

        # ═══════════════════════════════════════════════════════
        # LAYER 2 — Did-you-mean (partial match, 0.30–0.55)
        # ═══════════════════════════════════════════════════════
        if retrieval.top_score >= RAG.DID_YOU_MEAN_THRESHOLD:
            return self._did_you_mean(
                retrieval=retrieval,
                query_embedding=query_embedding,
                message=message,
                locale=locale,
            )

        # ═══════════════════════════════════════════════════════
        # LAYER 2.5 — Fallback generation (on-domain only)
        # ═══════════════════════════════════════════════════════
        if is_on_domain(message) and not is_off_domain(message):
            return await self._fallback_generation(
                message=message,
                locale=locale,
                retrieval=retrieval,
                query_embedding=query_embedding,
            )

        # ═══════════════════════════════════════════════════════
        # LAYER 6 — Refusal (off-domain)
        # ═══════════════════════════════════════════════════════
        return _PipelineResult(
            answer=refusal_for_category(retrieval.category_hint),
            source_cards=[],
            refused=True,
            is_welcome=False,
            cached=False,
            retrieval_top_score=retrieval.top_score,
            chunk_ids=[],
        )

    # ═══════════════════════════════════════════════════
    # Answer paths
    # ═══════════════════════════════════════════════════
    async def _answer_from_retrieval(
        self,
        *,
        message: str,
        locale: str,
        retrieval,
        query_embedding: list[float],
    ) -> _PipelineResult:
        if await self._daily_budget_exceeded():
            return _PipelineResult(
                answer=RAG_BUDGET_EXCEEDED_MESSAGE,
                source_cards=[],
                refused=True,
                is_welcome=False,
                cached=False,
                retrieval_top_score=retrieval.top_score,
                chunk_ids=[c.chunk_id for c in retrieval.chunks],
            )

        context = self._build_context(retrieval.chunks)
        system_prompt = build_system_prompt(
            context=context,
            category_hint=retrieval.category_hint,
        )

        raw_answer = await self._generate(
            system_prompt=system_prompt,
            user_message=message,
        )
        if raw_answer is None:
            return _PipelineResult(
                answer=RAG_UNAVAILABLE_MESSAGE,
                source_cards=[],
                refused=True,
                is_welcome=False,
                cached=False,
                retrieval_top_score=retrieval.top_score,
                chunk_ids=[c.chunk_id for c in retrieval.chunks],
            )

        out_verdict = self.output_guard.check(
            answer=raw_answer, chunks=retrieval.chunks
        )
        if out_verdict.is_refusal:
            return _PipelineResult(
                answer=out_verdict.answer,
                source_cards=[],
                refused=True,
                is_welcome=False,
                cached=False,
                retrieval_top_score=retrieval.top_score,
                chunk_ids=[c.chunk_id for c in retrieval.chunks],
            )

        source_cards = self._build_source_cards(
            answer=out_verdict.answer, chunks=retrieval.chunks
        )
        result = _PipelineResult(
            answer=out_verdict.answer,
            source_cards=source_cards,
            refused=False,
            is_welcome=False,
            cached=False,
            retrieval_top_score=retrieval.top_score,
            chunk_ids=[c.chunk_id for c in retrieval.chunks],
        )
        if (
            not out_verdict.stripped
            and retrieval.top_score >= 0.7
            and query_embedding is not None
        ):
            await self._safe_cache_store(
                message=message,
                locale=locale,
                payload=result,
                query_embedding=query_embedding,
            )
        return result

    def _did_you_mean(
        self,
        *,
        retrieval,
        query_embedding: Optional[list[float]],
        message: str,
        locale: str,
    ) -> _PipelineResult:
        titles: list[str] = []
        seen: set[str] = set()
        for c in retrieval.chunks:
            title = (
                c.meta.get("title")
                or c.meta.get("canonical_name")
                or c.meta.get("name")
            )
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
                if len(titles) >= RAG.DID_YOU_MEAN_COUNT:
                    break

        if not titles:
            return _PipelineResult(
                answer=refusal_for_category(retrieval.category_hint),
                source_cards=[],
                refused=True,
                is_welcome=False,
                cached=False,
                retrieval_top_score=retrieval.top_score,
                chunk_ids=[],
            )

        bullets = "\n".join(f"- {t}" for t in titles)
        answer = (
            "I don't have a specific answer for that, but these "
            "articles might help:\n\n"
            f"{bullets}\n\n"
            "You can also browse the catalog or try rephrasing "
            "your question."
        )

        return _PipelineResult(
            answer=answer,
            source_cards=self._build_source_cards(
                answer=answer, chunks=retrieval.chunks
            ),
            refused=False,
            is_welcome=False,
            cached=False,
            retrieval_top_score=retrieval.top_score,
            chunk_ids=[c.chunk_id for c in retrieval.chunks],
        )

    async def _fallback_generation(
        self,
        *,
        message: str,
        locale: str,
        retrieval,
        query_embedding: Optional[list[float]],
    ) -> _PipelineResult:
        from app.core.rag_config import RAG_FALLBACK_SYSTEM_PROMPT

        if await self._daily_budget_exceeded():
            return _PipelineResult(
                answer=RAG_BUDGET_EXCEEDED_MESSAGE,
                source_cards=[],
                refused=True,
                is_welcome=False,
                cached=False,
                retrieval_top_score=retrieval.top_score,
                chunk_ids=[],
            )

        raw_answer = await self._generate(
            system_prompt=RAG_FALLBACK_SYSTEM_PROMPT,
            user_message=message,
        )
        if raw_answer is None:
            return _PipelineResult(
                answer=RAG_UNAVAILABLE_MESSAGE,
                source_cards=[],
                refused=True,
                is_welcome=False,
                cached=False,
                retrieval_top_score=retrieval.top_score,
                chunk_ids=[],
            )

        import re
        sentences = re.split(r"(?<=[.!?])\s+|\n+", raw_answer.strip())
        kept: list[str] = []
        for s in sentences:
            if re.search(
                r"\d+[.,]\d{2}\s*(DT|TND)", s, re.IGNORECASE
            ):
                continue
            if re.search(r"https?://", s, re.IGNORECASE):
                continue
            kept.append(s)
        filtered = " ".join(kept).strip()

        if not filtered or len(filtered.split()) < 10:
            return _PipelineResult(
                answer=refusal_for_category(retrieval.category_hint),
                source_cards=[],
                refused=True,
                is_welcome=False,
                cached=False,
                retrieval_top_score=retrieval.top_score,
                chunk_ids=[],
            )

        filtered += (
            "\n\nI don't have a specific article on this, so if "
            "you need exact details, try browsing the site or "
            "contacting support."
        )

        return _PipelineResult(
            answer=filtered,
            source_cards=[],
            refused=False,
            is_welcome=False,
            cached=False,
            retrieval_top_score=retrieval.top_score,
            chunk_ids=[],
        )

    # ═══════════════════════════════════════════════════
    # Welcome
    # ═══════════════════════════════════════════════════
    async def _welcome(
        self,
        *,
        session_id: str,
        user_id: Optional[UUID],
        locale: str,
        start: float,
    ) -> RagChatResponse:
        await self._mark_greeted(session_id)
        remaining = await self._remaining_anonymous(
            session_id=session_id, user_id=user_id
        )
        return RagChatResponse(
            answer=RAG_WELCOME_MESSAGE,
            source_cards=[],
            cached=True,
            refused=False,
            is_welcome=True,
            requires_auth=False,
            remaining_anonymous=remaining,
            session_id=session_id,
            latency_ms=self._ms_since(start),
            request_id=_new_request_id(),
        )

    # ═══════════════════════════════════════════════════
    # Pipeline helpers
    # ═══════════════════════════════════════════════════
    async def _embed(self, message: str) -> Optional[list[float]]:
        try:
            from app.core.embeddings import embed_text
            return await embed_text(message, db=self.db)
        except Exception as exc:
            logger.warning(
                "rag_embed_failed", error=str(exc)[:200]
            )
            return None

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for c in chunks:
            safe_text = c.text.replace("[", "(").replace("]", ")")
            parts.append(f"[{c.chunk_id}]\n{safe_text}")
        return "\n\n---\n\n".join(parts)

    async def _generate(
        self,
        *,
        system_prompt: str,
        user_message: str,
    ) -> Optional[str]:
        sem = _get_sem()
        acquired = False
        try:
            await asyncio.wait_for(sem.acquire(), timeout=5.0)
            acquired = True
        except asyncio.TimeoutError:
            logger.warning("rag_concurrency_timeout")
            return None

        try:
            for attempt in range(RAG.LLM_RETRY_ATTEMPTS + 1):
                try:
                    return await self.ollama.chat(
                        system_prompt=system_prompt,
                        user_message=user_message,
                        temperature=RAG.TEMPERATURE,
                        max_tokens=RAG.MAX_OUTPUT_TOKENS,
                        timeout_s=RAG.LLM_TIMEOUT_SECONDS,
                    )
                except _OllamaTimeout:
                    logger.warning(
                        "rag_llm_timeout", attempt=attempt
                    )
                    if attempt >= RAG.LLM_RETRY_ATTEMPTS:
                        return None
                except _OllamaError as exc:
                    logger.warning(
                        "rag_llm_error",
                        attempt=attempt,
                        error=str(exc)[:200],
                    )
                    if attempt >= RAG.LLM_RETRY_ATTEMPTS:
                        return None
            return None
        finally:
            if acquired:
                sem.release()

    def _build_source_cards(
        self,
        *,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[RagSourceCard]:
        mentioned: list[RetrievedChunk] = []
        mentioned_ids: set[str] = set()
        answer_lower = answer.lower()

        for c in chunks:
            if c.kind == "help":
                continue
            name = (
                c.meta.get("name")
                or c.meta.get("canonical_name")
                or ""
            ).lower()
            if name and name in answer_lower:
                mentioned.append(c)
                mentioned_ids.add(c.chunk_id)

        ordered = mentioned + [
            c
            for c in chunks
            if c.kind != "help" and c.chunk_id not in mentioned_ids
        ]

        cards: list[RagSourceCard] = []
        for c in ordered:
            if c.kind == "product":
                cards.append(RagSourceCard(
                    kind="product",
                    product_id=c.product_id,
                    name=c.meta.get("name") or "",
                    site=c.meta.get("site"),
                    price=(
                        c.meta.get("discount_price")
                        or c.meta.get("price")
                    ),
                    availability=c.meta.get("availability"),
                    main_image=c.meta.get("main_image"),
                    url=c.meta.get("url") or "",
                ))
            elif c.kind == "canonical":
                cards.append(RagSourceCard(
                    kind="canonical",
                    canonical_id=c.canonical_id,
                    name=c.meta.get("canonical_name") or "",
                    site_count=c.meta.get("site_count"),
                    cheapest_site=c.meta.get("cheapest_site"),
                    price=c.meta.get("cheapest_price"),
                ))
        return cards

    # ═══════════════════════════════════════════════════
    # Cache
    # ═══════════════════════════════════════════════════
    async def _safe_cache_lookup(
        self,
        *,
        message: str,
        locale: str,
        query_embedding: Optional[list[float]],
        anon_suggestion_only: bool,
    ) -> Optional[dict[str, Any]]:
        try:
            return await self.cache.lookup(
                message=message,
                locale=locale,
                query_embedding=query_embedding,
                anon_suggestion_only=anon_suggestion_only,
            )
        except Exception as exc:
            logger.warning(
                "rag_cache_lookup_failed", error=str(exc)[:200]
            )
            return None

    async def _safe_cache_store(
        self,
        *,
        message: str,
        locale: str,
        payload: _PipelineResult,
        query_embedding: Optional[list[float]],
    ) -> None:
        try:
            await self.cache.store(
                message=message,
                locale=locale,
                response={
                    "answer": payload.answer,
                    "source_cards": [
                        c.model_dump(mode="json")
                        for c in payload.source_cards
                    ],
                    "refused": payload.refused,
                    "is_welcome": payload.is_welcome,
                },
                query_embedding=query_embedding,
            )
        except Exception as exc:
            logger.warning(
                "rag_cache_store_failed", error=str(exc)[:200]
            )

    def _pipeline_from_cache(
        self,
        *,
        payload: dict[str, Any],
        cache_source: str,
    ) -> _PipelineResult:
        cards = [
            RagSourceCard.model_validate(c)
            for c in payload.get("source_cards", [])
        ]
        return _PipelineResult(
            answer=payload.get("answer", ""),
            source_cards=cards,
            refused=bool(payload.get("refused", False)),
            is_welcome=bool(payload.get("is_welcome", False)),
            cached=True,
            retrieval_top_score=0.0,
            chunk_ids=[],
            cache_source=cache_source,
        )

    # ═══════════════════════════════════════════════════
    # Anon counter
    # ═══════════════════════════════════════════════════
    async def _remaining_anonymous(
        self,
        *,
        session_id: str,
        user_id: Optional[UUID],
    ) -> Optional[int]:
        if user_id is not None:
            return None
        try:
            from app.core.cache import Cache
            key = RAG_ANON.COUNTER_KEY_PREFIX + session_id
            used = await Cache.get(key)
            if used is None:
                return RAG_ANON.QUESTION_LIMIT
            return max(0, RAG_ANON.QUESTION_LIMIT - int(used))
        except Exception as exc:
            logger.warning(
                "rag_anon_remaining_failed", error=str(exc)[:200]
            )
            return RAG_ANON.QUESTION_LIMIT

    async def _consume_anon_slot(
        self, *, session_id: str
    ) -> Optional[int]:
        try:
            from app.core.cache import Cache
            key = RAG_ANON.COUNTER_KEY_PREFIX + session_id
            used_raw = await Cache.get(key)
            used = int(used_raw) if used_raw is not None else 0
            new_used = used + 1
            ttl_seconds = RAG_ANON.COUNTER_TTL_HOURS * 3600
            await Cache.set(key, new_used, ttl_seconds)
            return max(0, RAG_ANON.QUESTION_LIMIT - new_used)
        except Exception as exc:
            logger.warning(
                "rag_anon_consume_failed", error=str(exc)[:200]
            )
            return None

    async def _mark_greeted(self, session_id: str) -> None:
        try:
            from app.core.cache import Cache
            key = RAG_ANON.GREETED_KEY_PREFIX + session_id
            await Cache.set(
                key, True, RAG_ANON.COUNTER_TTL_HOURS * 3600
            )
        except Exception:
            pass

    # ═══════════════════════════════════════════════════
    # Signal log
    # ═══════════════════════════════════════════════════
    async def _log_turn(
        self,
        *,
        user_id: Optional[UUID],
        session_id: str,
        message: str,
        result: _PipelineResult,
        latency_ms: float,
    ) -> None:
        if user_id is None:
            return

        hit_kind = (
            "cache" if result.cached
            else "refusal" if result.refused
            else "retrieval"
        )
        payload = {
            "query": normalize_message(message)[:200],
            "hit": hit_kind,
            "chunk_ids": result.chunk_ids[:20],
            "latency_ms": int(latency_ms),
            "cache_source": result.cache_source,
        }
        evt = UserEvent(
            user_id=user_id,
            event_type="rag_query",
            product_id=None,
            metadata_=payload,
            weight=3.0,
        )
        self.db.add(evt)
        await self.db.flush()

    # ═══════════════════════════════════════════════════
    # Budget
    # ═══════════════════════════════════════════════════
    async def _daily_budget_exceeded(self) -> bool:
        try:
            sql = text(
                """
                SELECT COUNT(*) FROM llm_usage
                WHERE endpoint LIKE 'rag%'
                AND created_at >= date_trunc('day', now())
                """
            )
            count = int((await self.db.execute(sql)).scalar_one())
            return count >= RAG.DAILY_CALL_BUDGET
        except Exception:
            return False

    # ═══════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════
    @staticmethod
    def _ms_since(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)


# ═══════════════════════════════════════════════════════
# Ollama client
# ═══════════════════════════════════════════════════════
class _OllamaTimeout(Exception):
    pass


class _OllamaError(Exception):
    pass


class _OllamaClient:
    def __init__(self, db=None) -> None:
        self.base_url = settings.ollama_base_url
        # Prefer rag_llm_model, fall back to ollama_model
        self.model = getattr(
            settings, "rag_llm_model",
            getattr(settings, "ollama_model", "llama3.1:8b"),
        )
        self._cost_tracker_db = db

    async def chat(
        self,
        *,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
    ) -> str:
        import time
        import httpx

        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=timeout_s
            ) as client:
                resp = await client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            raise _OllamaTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise _OllamaError(str(exc)) from exc
        except Exception as exc:
            raise _OllamaError(str(exc)) from exc

        latency_ms = int((time.monotonic() - t0) * 1000)

        msg = (data or {}).get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise _OllamaError("unexpected Ollama response shape")

        if self._cost_tracker_db is not None:
            try:
                from app.core.cost_tracker import CostTracker
                await CostTracker.record(
                    self._cost_tracker_db,
                    endpoint="rag_chat",
                    provider="ollama",
                    model=self.model,
                    prompt_tokens=int(
                        data.get("prompt_eval_count", 0)
                    ),
                    completion_tokens=int(
                        data.get("eval_count", 0)
                    ),
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                logger.warning(
                    "rag_cost_track_failed", error=str(exc)[:200]
                )

        return content.strip()


def _new_request_id() -> str:
    import uuid
    return str(uuid.uuid4())