# app/services/recommendation_verifier.py
"""
LLM verifier for one user's materialized recommendation feed.

Given a user_id, loads:
  - user_profiles (recent_searches, preferred_categories, etc.)
  - wishlist_items (up to 5 most recent)
  - recommendations (top 10 by rank)

Builds a compact prompt, asks Ollama "do these match the user's
intent?", parses the verdict.

The caller (the Celery task) decides what to do:
  - confirmed  -> no action
  - mismatch   -> log, optionally rebuild the feed
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user_profile import UserProfile
from app.models.wishlist import WishlistItem
from app.services.verification_prompts import (
    RECOMMENDATION_SYSTEM_PROMPT,
)

logger = get_logger(__name__)


@dataclass
class RecommendationVerdict:
    decision: str
    confidence: float = 0.0
    reason: str = ""
    mismatched_product_ids: list[UUID] = field(default_factory=list)
    dominant_signal: str = "unknown"
    llm_latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_model: str = ""
    item_count: int = 0


class RecommendationVerifier:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.base_url = settings.ollama_base_url
        self.model = getattr(
            settings, "rag_llm_model",
            getattr(settings, "ollama_model", "llama3.1:8b"),
        )

    async def verify(
        self,
        *,
        user_id: UUID,
        timeout_s: int = 120,
    ) -> RecommendationVerdict:
        # ── Profile
        profile = await self.db.get(UserProfile, user_id)
        if profile is None:
            return RecommendationVerdict(
                decision="error",
                reason="no profile for user",
            )

        # ── Recommendations (top 15, we'll trim to 10 in the payload)
        recs = list((await self.db.execute(
            select(Recommendation)
            .where(Recommendation.user_id == user_id)
            .order_by(Recommendation.rank.asc())
            .limit(15)
        )).scalars().all())

        if len(recs) < 5:
            return RecommendationVerdict(
                decision="error",
                reason="fewer than 5 recommendations to verify",
                item_count=len(recs),
            )

        product_ids = [r.product_id for r in recs]
        products = {
            p.id: p
            for p in (await self.db.execute(
                select(Product).where(Product.id.in_(product_ids))
            )).scalars().all()
        }

        # ── Wishlist (top 10, we'll trim to 5 in the payload)
        wishlist_rows = (await self.db.execute(
            select(Product)
            .join(WishlistItem, WishlistItem.product_id == Product.id)
            .where(WishlistItem.user_id == user_id)
            .order_by(WishlistItem.created_at.desc())
            .limit(10)
        )).scalars().all()

        payload = self._build_payload(
            profile, recs, products, wishlist_rows,
        )

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=timeout_s,
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={
                        "model": self.model,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 150,
                            "num_ctx": 2048,
                        },
                        "messages": [
                            {
                                "role": "system",
                                "content": RECOMMENDATION_SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    payload, separators=(",", ":"),
                                ),
                            },
                        ],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning(
                "rec_verify_llm_failed",
                user_id=str(user_id),
                error=f"{type(exc).__name__}: {str(exc)[:180]}",
            )
            return RecommendationVerdict(
                decision="error",
                reason=f"LLM call failed: {type(exc).__name__}",
                llm_latency_ms=int((time.monotonic() - t0) * 1000),
                llm_model=self.model,
                item_count=len(recs),
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        content = (data.get("message") or {}).get("content", "")
        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))

        parsed = self._parse_verdict(content, recs)
        parsed.llm_latency_ms = latency_ms
        parsed.prompt_tokens = prompt_tokens
        parsed.completion_tokens = completion_tokens
        parsed.llm_model = self.model
        parsed.item_count = len(recs)
        return parsed

    # ═══════════════════════════════════════════════════
    # Payload — minimal, ~500 tokens
    # ═══════════════════════════════════════════════════
    @staticmethod
    def _build_payload(
        profile, recs, products, wishlist_rows,
    ) -> dict:
        def _short(s: Optional[str], n: int = 40) -> Optional[str]:
            return s[:n] if s else None

        return {
            "wishlist": [
                {
                    "id": str(p.id),
                    "cat": _short(p.category, 30),
                    "brand": _short(p.brand, 20),
                    "name": _short(p.name, 40),
                }
                for p in wishlist_rows[:5]
            ],
            "searches": (profile.recent_searches or [])[:5],
            "recs": [
                {
                    "id": str(r.product_id),
                    "name": _short(
                        products.get(r.product_id).name
                        if r.product_id in products else None,
                        40,
                    ),
                    "cat": _short(
                        products.get(r.product_id).category
                        if r.product_id in products else None,
                        30,
                    ),
                    "brand": _short(
                        products.get(r.product_id).brand
                        if r.product_id in products else None,
                        20,
                    ),
                }
                for r in recs[:10]
            ],
        }

    # ═══════════════════════════════════════════════════
    # Verdict parsing
    # ═══════════════════════════════════════════════════
    def _parse_verdict(
        self, raw: str, recs: list[Recommendation],
    ) -> RecommendationVerdict:
        text_body = raw.strip()
        if text_body.startswith("```"):
            text_body = re.sub(r"^```[a-z]*\n?", "", text_body)
            text_body = re.sub(r"\n?```$", "", text_body)

        try:
            data = json.loads(text_body)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text_body, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if not isinstance(data, dict):
            logger.warning(
                "rec_verify_bad_json",
                raw_prefix=text_body[:120],
            )
            return RecommendationVerdict(
                decision="mismatch",
                confidence=0.0,
                reason="LLM returned unparseable JSON",
            )

        matches = bool(data.get("matches_intent", False))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        reason = str(data.get("reason", ""))[:400]
        signal = str(data.get("dominant_signal", "unknown"))

        real_ids = {r.product_id for r in recs}
        raw_mism = data.get("mismatched_product_ids") or []
        mismatched: list[UUID] = []
        for x in raw_mism:
            try:
                uid = UUID(str(x))
            except (ValueError, TypeError):
                continue
            if uid in real_ids:
                mismatched.append(uid)

        if matches and not mismatched:
            return RecommendationVerdict(
                decision="confirmed",
                confidence=confidence,
                reason=reason,
                dominant_signal=signal,
            )
        return RecommendationVerdict(
            decision="mismatch",
            confidence=confidence,
            reason=reason,
            mismatched_product_ids=mismatched,
            dominant_signal=signal,
        )