# app/services/profile_service.py
"""Aggregate user_events into user_profiles with tier weighting."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.core.recommendation_config import (
    COLD_START_MIN_DISTINCT_PRODUCTS,
    COLD_START_MIN_EVENTS,
    PROFILE,
    TIER,
    tier_multiplier_h,
    tier_name_h,
)
from app.models.product import Product
from app.models.user_events import UserEvent
from app.repositories.profile_repo import ProfileRepo

logger = get_logger(__name__)


class ProfileService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = ProfileRepo(db)

    async def rebuild(self, user_id: UUID) -> None:
        """
        Rebuild the profile for one user from their recent event log.

        Each event contributes
            base_weight * tier_multiplier(age)
        so a wishlist from 2h ago counts ~3x a wishlist from a week
        ago, and ~10x a wishlist from a month ago. Category and
        brand affinity are tagged with a tier so the ranker can
        distinguish "today's interest" from "long-term taste".
        """
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=PROFILE.EVENT_WINDOW_DAYS)
        expires_at = now + timedelta(hours=PROFILE.TTL_HOURS)

        result = await self.db.execute(
            select(UserEvent).where(
                UserEvent.user_id == user_id,
                UserEvent.created_at >= since,
            )
        )
        events = list(result.scalars().all())

        if not events:
            await self.repo.delete(user_id)
            await self.db.commit()
            logger.info(
                "profile_dropped_no_recent_events",
                user_id=str(user_id),
                window_days=PROFILE.EVENT_WINDOW_DAYS,
            )
            return

        # ── Tier-weighted product counter. Search events have no
        #    product_id and are handled separately below.
        events_by_pid: dict[UUID, list[UserEvent]] = {}
        product_weights: Counter[UUID] = Counter()
        last_seen = None

        for e in events:
            if last_seen is None or e.created_at > last_seen:
                last_seen = e.created_at
            if e.product_id is None:
                continue
            age_h = (now - e.created_at).total_seconds() / 3600.0
            mult = tier_multiplier_h(age_h)
            base = float(e.weight or 1.0)
            product_weights[e.product_id] += base * mult
            events_by_pid.setdefault(e.product_id, []).append(e)

        distinct = len(product_weights)

        # ── Fetch product metadata.
        product_rows = []
        if product_weights:
            stmt = select(
                Product.id,
                Product.category,
                Product.brand,
                Product.price,
                Product.discount_price,
            ).where(Product.id.in_(list(product_weights.keys())))
            product_rows = (await self.db.execute(stmt)).all()

        # ── Aggregate into fresh/long buckets.
        cat_fresh: Counter[str] = Counter()
        cat_long: Counter[str] = Counter()
        brand_fresh: Counter[str] = Counter()
        brand_long: Counter[str] = Counter()
        prices: list[float] = []

        for pid, cat, brand, price, discount in product_rows:
            w = product_weights.get(pid, 1.0)
            pid_events = events_by_pid.get(pid) or []
            if pid_events:
                freshest_age_h = min(
                    (now - e.created_at).total_seconds() / 3600.0
                    for e in pid_events
                )
            else:
                freshest_age_h = 9999.0
            mult = tier_multiplier_h(freshest_age_h)
            is_fresh = mult >= TIER.TODAY_MULT

            if cat:
                if is_fresh:
                    cat_fresh[cat] += w
                else:
                    cat_long[cat] += w
            if brand:
                if is_fresh:
                    brand_fresh[brand] += w
                else:
                    brand_long[brand] += w

            effective = (
                float(discount)
                if discount is not None and float(discount) > 0
                else float(price)
                if price is not None
                else None
            )
            if effective is not None and effective > 0:
                prices.append(effective)

        # ── Merge categories, one entry per name with a tier tag.
        cat_merged: dict[str, dict] = {}
        for name, w in cat_fresh.items():
            cat_merged[name] = {
                "name": name, "weight": round(w, 2), "tier": "fresh",
            }
        for name, w in cat_long.items():
            if name in cat_merged:
                cat_merged[name]["weight"] = round(
                    cat_merged[name]["weight"] + w, 2
                )
                cat_merged[name]["tier"] = "mixed"
            else:
                cat_merged[name] = {
                    "name": name, "weight": round(w, 2), "tier": "long",
                }
        top_categories = sorted(
            cat_merged.values(), key=lambda x: x["weight"], reverse=True
        )[:8]

        brand_merged: dict[str, dict] = {}
        for name, w in brand_fresh.items():
            brand_merged[name] = {
                "name": name, "weight": round(w, 2), "tier": "fresh",
            }
        for name, w in brand_long.items():
            if name in brand_merged:
                brand_merged[name]["weight"] = round(
                    brand_merged[name]["weight"] + w, 2
                )
                brand_merged[name]["tier"] = "mixed"
            else:
                brand_merged[name] = {
                    "name": name, "weight": round(w, 2), "tier": "long",
                }
        top_brands = sorted(
            brand_merged.values(), key=lambda x: x["weight"], reverse=True
        )[:8]

        # ── Price band.
        price_min = price_max = price_mean = p25 = p75 = None
        if prices:
            prices_sorted = sorted(prices)
            n = len(prices_sorted)
            price_min = round(prices_sorted[0], 2)
            price_max = round(prices_sorted[-1], 2)
            price_mean = round(sum(prices_sorted) / n, 2)
            p25 = round(prices_sorted[max(0, n // 4)], 2)
            p75 = round(prices_sorted[min(n - 1, (n * 3) // 4)], 2)

        is_cold = (
            distinct < COLD_START_MIN_DISTINCT_PRODUCTS
            or len(events) < COLD_START_MIN_EVENTS
        )

        # ── Recent searches, freshest first, deduped.
        recent_searches: list[str] = []
        for e in sorted(
            events, key=lambda x: x.created_at, reverse=True
        ):
            if e.event_type == "search":
                q = (e.metadata_ or {}).get("query")
                if q and q not in recent_searches:
                    recent_searches.append(q)
                    if len(recent_searches) >= 10:
                        break

        await self.repo.upsert(
            user_id,
            total_events=len(events),
            distinct_products=distinct,
            preferred_categories=top_categories,
            preferred_brands=top_brands,
            price_min=price_min,
            price_max=price_max,
            price_mean=price_mean,
            price_p25=p25,
            price_p75=p75,
            last_event_at=last_seen,
            is_cold_start=is_cold,
            recent_searches=recent_searches,
            expires_at=expires_at,
        )
        await self.db.commit()

        logger.info(
            "profile_rebuilt",
            user_id=str(user_id),
            events=len(events),
            fresh_categories=len(cat_fresh),
            long_categories=len(cat_long),
            cold_start=is_cold,
            window_days=PROFILE.EVENT_WINDOW_DAYS,
            expires_at=expires_at.isoformat(),
        )