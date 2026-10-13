# app/services/ranking_service.py
"""Stage 2 — deterministic ranking with tier weighting and priority bands."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.core.recommendation_config import (
    PROFILE,
    RANKING,
    SELLABLE_AVAILABILITY,
    TIER,
    profile_tier_multiplier,
    tier_multiplier_h,
)
from app.models.product import Product
from app.models.user_profile import UserProfile

logger = get_logger(__name__)


@dataclass
class ScoredCandidate:
    product_id: UUID
    score: float
    dominant_signal: str
    content_sim: float
    price_affinity: float
    brand_affinity: float
    category_affinity: float
    popularity: float
    recency: float
    seen_penalty: float
    search_boost: float = 0.0
    priority_boost: float = 0.0


class RankingService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model = settings.active_embedding_model

    async def rank(
        self,
        *,
        candidates: list[UUID],
        profile: Optional[UserProfile],
        seen_products: set[UUID],
        priority_ids: Optional[dict[str, set[UUID]]] = None,
        top_k: int = RANKING.STAGE2_TOP_K,
    ) -> list[ScoredCandidate]:
        if not candidates:
            return []

        priority_ids = priority_ids or {}

        # ── Load product rows.
        stmt = select(Product).where(Product.id.in_(candidates))
        products = list((await self.db.execute(stmt)).scalars().all())
        sellable_set = set(SELLABLE_AVAILABILITY)
        products = [
            p for p in products
            if (p.availability or "") in sellable_set
        ]
        product_map = {p.id: p for p in products}

        # ── Content similarity to the user's recent seeds.
        seed_weights = await self._seed_recency_weights(
            seed_ids=list(seen_products)[:20]
        )
        sims = await self._content_similarities(
            candidate_ids=candidates,
            seed_weights=seed_weights,
        )

        # ── Tier-aware category / brand affinity.
        top_brands: dict[str, float] = {}
        top_categories: dict[str, tuple[float, float]] = {}
        price_center = 0.0

        if profile is not None:
            for entry in profile.preferred_brands or []:
                if isinstance(entry, dict) and entry.get("name"):
                    name = str(entry["name"]).lower()
                    w = float(entry.get("weight") or 1.0)
                    tier = entry.get("tier", "long")
                    top_brands[name] = w * profile_tier_multiplier(tier)

            for entry in profile.preferred_categories or []:
                if isinstance(entry, dict) and entry.get("name"):
                    name = str(entry["name"]).lower()
                    w = float(entry.get("weight") or 1.0)
                    tier = entry.get("tier", "long")
                    top_categories[name] = (w, profile_tier_multiplier(tier))

            price_center = float(
                profile.price_p75
                or profile.price_mean
                or 0.0
            )

        max_pop = max(
            (p.price_change_count or 0 for p in products), default=0
        ) or 1
        max_brand_w = max(top_brands.values(), default=1.0) or 1.0
        max_cat_w = max(
            (w * m for w, m in top_categories.values()),
            default=1.0,
        ) or 1.0

        # ── Fresh search terms for the literal-match boost.
        fresh_queries: list[str] = []
        if profile and profile.recent_searches:
            fresh_queries = [
                q.lower() for q in (profile.recent_searches or [])[:5]
                if q
            ]

        # ── Priority band sets, for fast membership tests.
        wishlist_set = priority_ids.get("wishlist", set())
        search_set = priority_ids.get("search", set())
        cart_set = priority_ids.get("cart", set())
        click_set = priority_ids.get("click", set())
        dwell_set = priority_ids.get("dwell", set())

        now = datetime.now(timezone.utc)
        scored: list[ScoredCandidate] = []

        for cid in candidates:
            p = product_map.get(cid)
            if p is None:
                continue

            content_sim = sims.get(cid, 0.0)

            brand = (p.brand or "").lower()
            brand_w = top_brands.get(brand, 0.0)
            brand_aff = (
                min(1.0, brand_w / max_brand_w) if brand_w else 0.0
            )

            cat = (p.category or "").lower()
            cat_w, cat_mult = top_categories.get(cat, (0.0, 1.0))
            if cat_w:
                category_aff = min(
                    1.0, (cat_w * cat_mult) / max_cat_w
                )
            else:
                category_aff = 0.0

            price = self._effective_price(p)
            if price and price_center > 0:
                log_p = math.log(price)
                log_c = math.log(price_center)
                price_aff = math.exp(-((log_p - log_c) ** 2) / 2)
            else:
                price_aff = 0.5

            pop = (p.price_change_count or 0) / max_pop
            recency = self._recency_score(p.last_updated_at, now=now)
            seen = 1.0 if cid in seen_products else 0.0

            # ── Search-literal boost.
            search_boost = 0.0
            if fresh_queries:
                p_name_l = (p.name or "").lower()
                p_cat_l = (p.category or "").lower()
                for q in fresh_queries:
                    if q in p_name_l:
                        search_boost = TIER.SEARCH_LITERAL_BOOST
                        break
                    if q in p_cat_l:
                        search_boost = max(
                            search_boost, TIER.SEARCH_CATEGORY_BOOST
                        )

            # ── Priority boost (wishlist > search > cart > click > dwell).
            priority_boost = 0.0
            if cid in wishlist_set:
                priority_boost = RANKING.PRIORITY_WISHLIST
            elif cid in search_set:
                priority_boost = RANKING.PRIORITY_SEARCH
            elif cid in cart_set:
                priority_boost = RANKING.PRIORITY_CART
            elif cid in click_set:
                priority_boost = RANKING.PRIORITY_CLICK
            elif cid in dwell_set:
                priority_boost = RANKING.PRIORITY_DWELL

            score = (
                RANKING.CONTENT * content_sim
                + RANKING.POPULARITY * pop
                + RANKING.PRICE_AFFINITY * price_aff
                + RANKING.BRAND_AFFINITY * brand_aff
                + RANKING.CATEGORY_AFFINITY * category_aff
                + RANKING.RECENCY * recency
                + search_boost
                + priority_boost
                - RANKING.SEEN_PENALTY * seen
            )

            contributions = {
                "content": RANKING.CONTENT * content_sim,
                "price": RANKING.PRICE_AFFINITY * price_aff,
                "brand": RANKING.BRAND_AFFINITY * brand_aff,
                "category": RANKING.CATEGORY_AFFINITY * category_aff,
                "popularity": RANKING.POPULARITY * pop,
                "search": search_boost,
                "priority": priority_boost,
            }
            dominant = max(contributions, key=contributions.get)

            scored.append(ScoredCandidate(
                product_id=cid,
                score=score,
                dominant_signal=dominant,
                content_sim=content_sim,
                price_affinity=price_aff,
                brand_affinity=brand_aff,
                category_affinity=category_aff,
                popularity=pop,
                recency=recency,
                seen_penalty=seen,
                search_boost=search_boost,
                priority_boost=priority_boost,
            ))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    # ═══════════════════════════════════════════════════════
    # Content similarity with tier-weighted seeds
    # ═══════════════════════════════════════════════════════
    async def _seed_recency_weights(
        self, *, seed_ids: list[UUID]
    ) -> dict[UUID, float]:
        if not seed_ids:
            return {}

        now = datetime.now(timezone.utc)
        since = now - timedelta(days=PROFILE.EVENT_WINDOW_DAYS)

        sql = text(
            """
            SELECT product_id,
                   MAX(weight) AS event_weight,
                   MIN(EXTRACT(EPOCH FROM (:now - created_at)) / 3600.0)
                       AS freshest_age_h
            FROM user_events
            WHERE product_id = ANY(:seed_ids)
              AND created_at >= :since
            GROUP BY product_id
            """
        )
        try:
            rows = (await self.db.execute(sql, {
                "seed_ids": [str(x) for x in seed_ids],
                "since": since,
                "now": now,
            })).all()
        except Exception:
            return {sid: 1.0 for sid in seed_ids}

        out: dict[UUID, float] = {}
        for pid, event_weight, freshest_age_h in rows:
            if pid is None:
                continue
            mult = tier_multiplier_h(float(freshest_age_h or 999.0))
            out[pid] = float(event_weight or 1.0) * mult

        for sid in seed_ids:
            out.setdefault(sid, 0.1)
        return out

    async def _content_similarities(
        self,
        *,
        candidate_ids: list[UUID],
        seed_weights: dict[UUID, float],
    ) -> dict[UUID, float]:
        if not seed_weights or not candidate_ids:
            return {}

        sql = text(
            """
            SELECT pe1.product_id AS seed_id,
                pe2.product_id AS candidate_id,
                1 - (pe1.embedding <=> pe2.embedding) AS similarity
            FROM product_embeddings pe1
            JOIN product_embeddings pe2
            ON pe2.model = pe1.model
            AND pe2.product_id <> pe1.product_id
            WHERE pe1.product_id = :seed_id
            AND pe1.model = :model
            AND pe2.product_id = ANY(:candidate_ids)
            AND pe2.model = :model
            """
        )

        weighted_sum: dict[UUID, float] = {}
        weight_total: dict[UUID, float] = {}

        for seed_id, w in seed_weights.items():
            try:
                rows = (await self.db.execute(sql, {
                    "seed_id": str(seed_id),
                    "candidate_ids": [str(x) for x in candidate_ids],
                    "model": self.model,
                })).all()
            except Exception as exc:
                logger.warning(
                    "content_similarity_seed_failed",
                    seed_id=str(seed_id),
                    error=str(exc)[:200],
                )
                try:
                    await self.db.rollback()
                except Exception:
                    pass
                continue

            for _seed, candidate_id, similarity in rows:
                if candidate_id is None or similarity is None:
                    continue
                weighted_sum[candidate_id] = (
                    weighted_sum.get(candidate_id, 0.0)
                    + float(w) * float(similarity)
                )
                weight_total[candidate_id] = (
                    weight_total.get(candidate_id, 0.0) + float(w)
                )

        return {
            cid: weighted_sum[cid] / weight_total[cid]
            for cid in weighted_sum
            if weight_total.get(cid, 0.0) > 0
        }

    # ═══════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _recency_score(
        last_updated_at: Optional[datetime],
        *,
        now: datetime,
    ) -> float:
        if last_updated_at is None:
            return 0.0
        if last_updated_at.tzinfo is None:
            last_updated_at = last_updated_at.replace(
                tzinfo=timezone.utc
            )
        age_days = (now - last_updated_at).total_seconds() / 86400.0
        window = PROFILE.EVENT_WINDOW_DAYS
        if age_days <= 0:
            return 1.0
        if age_days >= window:
            return 0.0
        return 1.0 - (age_days / window)

    @staticmethod
    def _effective_price(p: Product) -> Optional[float]:
        if p.price is None:
            return None
        if p.discount_price is not None and float(p.discount_price) > 0:
            return float(p.discount_price)
        return float(p.price)