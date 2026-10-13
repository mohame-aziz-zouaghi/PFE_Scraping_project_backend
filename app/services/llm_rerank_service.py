"""Stage 3 — LLM rerank + rationale."""
from __future__ import annotations

import asyncio
import random
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.llm_client import LLMClient
from app.core.logging_config import get_logger
from app.core.recommendation_config import (
    LLM_RERANK,
    RECS_RERANK_CACHE_NAMESPACE,
    RECS_RERANK_PROMPT_VERSION,
)
from app.models.product import Product
from app.models.user_profile import UserProfile
from app.services.ranking_service import ScoredCandidate

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Schema the LLM must return
# ═══════════════════════════════════════════════════════
class RerankedItem(BaseModel):
    product_id: UUID
    rank: int = Field(ge=1, le=30)
    reason: str = Field(min_length=5, max_length=300)


class RerankOutput(BaseModel):
    recommendations: list[RerankedItem]


class LLMRerankService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self._sem = asyncio.Semaphore(LLM_RERANK.MAX_CONCURRENCY)

    async def rerank(
        self,
        *,
        user_id: UUID,
        profile: Optional[UserProfile],
        candidates: list[ScoredCandidate],
        products: dict[UUID, Product],
        recent_view_names: list[str],
        final_limit: int = 12,
    ) -> Optional[list[RerankedItem]]:
        """Return reranked items, or None on failure (caller falls
        back to Stage 2 order)."""

        if not candidates:
            return None

        # ── Build the payload the LLM sees
        candidate_payload = []
        for i, s in enumerate(candidates[:30]):
            p = products.get(s.product_id)
            if p is None:
                continue
            candidate_payload.append({
                "rank": i + 1,
                "product_id": str(s.product_id),
                "name": p.name or "Untitled",
                "brand": p.brand,
                "category": p.category,
                "price": (
                    float(p.discount_price)
                    if p.discount_price is not None
                    and float(p.discount_price) > 0
                    else float(p.price)
                    if p.price is not None
                    else None
                ),
                "stage2_score": round(s.score, 4),
                "dominant_signal": s.dominant_signal,
            })

        if not candidate_payload:
            return None

        profile_payload = {
            "preferred_categories": (
                profile.preferred_categories[:3] if profile else []
            ),
            "preferred_brands": (
                profile.preferred_brands[:3] if profile else []
            ),
            "price_range": (
                [
                    float(profile.price_p25)
                    if profile.price_p25 is not None else None,
                    float(profile.price_p75)
                    if profile.price_p75 is not None else None,
                ]
                if profile else [None, None]
            ),
            "recent_views": recent_view_names[:5],
            "cold_start": (
                profile.is_cold_start if profile else True
            ),
        }

        system = (
            "You are a product recommendation explainer for a Tunisian "
            "e-commerce platform. You receive a user profile and up to 30 "
            "candidate products already ranked by a deterministic scorer. "
            "Return the top "
            f"{final_limit} in the same order unless you have a strong "
            "reason to promote an item by at most "
            f"{LLM_RERANK.MAX_REORDER} positions. "
            "For each item, write a one-line reason (max 140 chars) that "
            "references a specific fact from the profile: a category "
            "match, a brand match, a price fit, or similarity to a "
            "specific recently viewed item. Never invent features. "
            "Reply with JSON only: "
            '{"recommendations": [{"product_id": "uuid", "rank": 1, '
            '"reason": "..."}]}'
        )

        payload = {
            "user_profile": profile_payload,
            "candidates": candidate_payload,
        }

        cache_key_parts = [
            RECS_RERANK_CACHE_NAMESPACE,
            str(user_id),
            self._candidate_set_hash(candidate_payload),
        ]

        client = LLMClient(
            db=self.db,
            endpoint="recs_rerank",
            prompt_version=RECS_RERANK_PROMPT_VERSION,
        )

        last_exc: Optional[Exception] = None
        for attempt in range(LLM_RERANK.RETRY_ATTEMPTS):
            try:
                async with self._sem:
                    result = await client.chat_json(
                        system=system,
                        user=payload,
                        schema=RerankOutput,
                        cache_key_parts=cache_key_parts,
                        cache_ttl_seconds=LLM_RERANK.CACHE_TTL_SECONDS,
                    )
                # Enforce ordering invariant
                return self._enforce_order(result, candidates)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "recs_rerank_attempt_failed",
                    attempt=attempt + 1,
                    user_id=str(user_id),
                    error=str(exc)[:200],
                )
                if attempt + 1 < LLM_RERANK.RETRY_ATTEMPTS:
                    await asyncio.sleep(
                        LLM_RERANK.RETRY_JITTER_SECONDS
                        * (0.5 + random.random())
                    )

        logger.warning(
            "recs_rerank_failed",
            user_id=str(user_id),
            error=str(last_exc)[:200] if last_exc else "unknown",
        )
        return None

    # ═══════════════════════════════════════════════════════
    # Internals
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _candidate_set_hash(payload: list[dict]) -> str:
        import hashlib
        ids = sorted(str(c["product_id"]) for c in payload)
        return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]

    @staticmethod
    def _enforce_order(
        result: RerankOutput,
        candidates: list[ScoredCandidate],
    ) -> list[RerankedItem]:
        """Hard backstop: if the LLM reorders a candidate by more
        than MAX_REORDER positions relative to Stage 2, clamp it."""
        stage2_index = {
            c.product_id: i for i, c in enumerate(candidates[:30])
        }
        cleaned: list[RerankedItem] = []
        for item in result.recommendations:
            if item.product_id not in stage2_index:
                continue
            original_idx = stage2_index[item.product_id]
            new_idx = item.rank - 1
            clamped_idx = max(
                original_idx - LLM_RERANK.MAX_REORDER,
                min(original_idx + LLM_RERANK.MAX_REORDER, new_idx),
            )
            cleaned.append(
                item.model_copy(update={"rank": clamped_idx + 1})
            )
        # Sort by (possibly clamped) rank
        cleaned.sort(key=lambda x: x.rank)
        # Deduplicate ranks while preserving order
        seen_products: set[UUID] = set()
        final: list[RerankedItem] = []
        for i, item in enumerate(cleaned):
            if item.product_id in seen_products:
                continue
            seen_products.add(item.product_id)
            final.append(item.model_copy(update={"rank": i + 1}))
        return final