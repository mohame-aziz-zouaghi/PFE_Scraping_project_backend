# app/services/comparison_service.py
"""
Read-side service for /compare endpoints.

Trusts the denormalized cheapest_price columns written by the matching
pipeline. Live aggregation only for most_expensive + savings, which are
not denormalized.

The effective-price formula lives in matching_helpers.EFFECTIVE_PRICE_SQL
so it can never diverge from the matching pass.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import NotFoundError
from app.core.logging_config import get_logger
from app.repositories.canonical_repo import CanonicalRepo
from app.schemas.compare import (
    BaseGroupDetail,
    GroupDetail,
    GroupHistoryPoint,
    GroupHistoryResponse,
    GroupListing,
    ProductGroupResponse,
    VariantInBase,
)
from app.services.matching_helpers import (
    EFFECTIVE_PRICE_SQL,
    effective_price,
)

logger = get_logger(__name__)


class ComparisonService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = CanonicalRepo(db)

    # ═══════════════════════════════════════════════════════
    # Canonical (variant) detail
    # ═══════════════════════════════════════════════════════
    async def get_group_detail(self, group_id: UUID) -> GroupDetail:
        group = await self.repo.get(group_id)
        if not group:
            raise NotFoundError("Canonical product not found")

        listings = await self.repo.get_listings(group_id)
        listing_items = [_to_listing(p) for p in listings]

        cheapest_price = (
            float(group.cheapest_price)
            if group.cheapest_price is not None else None
        )
        cheapest_site = group.cheapest_site

        # Live most-expensive + defensive savings.
        # If the rollup is stale (cheapest > most_exp), treat the
        # live value as authoritative and recompute savings to zero
        # rather than serving a negative number.
        _, _, most_expensive = _summarize(listings)

        savings_amount = None
        savings_percent = None
        if (cheapest_price is not None
                and most_expensive is not None
                and most_expensive > cheapest_price):
            savings_amount = round(most_expensive - cheapest_price, 2)
            savings_percent = round(
                (savings_amount / most_expensive) * 100, 2
            )

        return GroupDetail(
            group_id=group.id,
            canonical_name=group.canonical_name,
            canonical_brand=group.canonical_brand,
            canonical_category=group.canonical_category,
            variant_signature=group.identity_signature,
            product_count=group.listing_count,
            site_count=group.site_count,
            confidence=float(group.confidence or 0),
            base_group_id=group.base_group_id,
            cheapest_site=cheapest_site,
            cheapest_price=cheapest_price,
            most_expensive_price=most_expensive,
            savings_amount=savings_amount,
            savings_percent=savings_percent,
            listings=listing_items,
        )

    # ═══════════════════════════════════════════════════════
    # Base detail
    # ═══════════════════════════════════════════════════════
    async def get_base_group_detail(
        self, base_id: UUID
    ) -> BaseGroupDetail:
        base = await self.repo.get_base(base_id)
        if not base:
            raise NotFoundError("Base group not found")

        variants = await self.repo.get_variants(base_id)

        variant_items: list[VariantInBase] = []
        for v in variants:
            v_price = (
                float(v.cheapest_price)
                if v.cheapest_price is not None else None
            )
            variant_items.append(
                VariantInBase(
                    group_id=v.id,
                    canonical_name=v.canonical_name,
                    variant_signature=v.identity_signature,
                    product_count=v.listing_count,
                    site_count=v.site_count,
                    cheapest_price=v_price,
                    cheapest_site=v.cheapest_site,
                )
            )

        # Trust the base rollup. Fallback only if the rollup is missing
        # (fresh row that hasn't run through matching yet).
        overall_cheapest_price: Optional[float] = None
        overall_cheapest_site: Optional[str] = None
        if base.cheapest_price is not None:
            overall_cheapest_price = float(base.cheapest_price)
            overall_cheapest_site = base.cheapest_site
        else:
            for v in variants:
                if v.cheapest_price is None:
                    continue
                v_price = float(v.cheapest_price)
                if (overall_cheapest_price is None
                        or v_price < overall_cheapest_price):
                    overall_cheapest_price = v_price
                    overall_cheapest_site = v.cheapest_site

        return BaseGroupDetail(
            base_group_id=base.id,
            base_name=base.base_name,
            base_brand=base.base_brand,
            base_category=base.base_category,
            variant_count=base.variant_count,
            site_count=base.site_count,
            total_product_count=base.listing_count,
            overall_cheapest_price=overall_cheapest_price,
            overall_cheapest_site=overall_cheapest_site,
            variants=variant_items,
        )

    # ═══════════════════════════════════════════════════════
    # Product → group resolution
    # ═══════════════════════════════════════════════════════
    async def get_product_group(
        self, product_id: UUID
    ) -> ProductGroupResponse:
        cid = await self.repo.get_for_product(product_id)
        if not cid:
            return ProductGroupResponse(
                product_id=product_id,
                group_id=None,
                base_group_id=None,
                canonical_name=None,
                site_count=0,
                listings=[],
            )
        detail = await self.get_group_detail(cid)
        return ProductGroupResponse(
            product_id=product_id,
            group_id=detail.group_id,
            base_group_id=detail.base_group_id,
            canonical_name=detail.canonical_name,
            site_count=detail.site_count,
            listings=detail.listings,
        )

    # ═══════════════════════════════════════════════════════
    # Price history
    # ═══════════════════════════════════════════════════════
    async def get_group_history(
        self, group_id: UUID, period_days: int = 30
    ) -> GroupHistoryResponse:
        group = await self.repo.get(group_id)
        if not group:
            raise NotFoundError("Canonical product not found")
        period_days = max(1, min(int(period_days), 3650))
        since = datetime.now(timezone.utc) - timedelta(days=period_days)

        sql = text(
            f"""
            SELECT
                DATE(ph.recorded_at) AS day,
                p.site,
                AVG({EFFECTIVE_PRICE_SQL})  AS avg_price,
                MIN({EFFECTIVE_PRICE_SQL})  AS min_price,
                COUNT(*) AS sample_count
            FROM price_history ph
            JOIN products p ON p.id = ph.product_id
            WHERE p.canonical_id = :cid
              AND ph.recorded_at >= :since
              AND ph.price IS NOT NULL
              AND ph.price > 0
              AND {EFFECTIVE_PRICE_SQL} > 0
            GROUP BY day, p.site
            ORDER BY day ASC, p.site ASC
            """
        )
        rows = (await self.db.execute(sql, {
            "cid": group_id, "since": since,
        })).all()
        points = [
            GroupHistoryPoint(
                date=r.day.isoformat(),
                site=r.site,
                avg_price=float(r.avg_price),
                min_price=float(r.min_price),
                sample_count=int(r.sample_count),
            )
            for r in rows
        ]
        return GroupHistoryResponse(
            group_id=group_id,
            canonical_name=group.canonical_name,
            period_days=period_days,
            points=points,
        )


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════
def _to_listing(p) -> GroupListing:
    return GroupListing(
        product_id=p.id,
        site=p.site,
        main_product_id=p.main_product_id,
        name=p.name or "",
        brand=p.brand,
        category=p.category,
        price=float(p.price) if p.price is not None else None,
        discount_price=(
            float(p.discount_price)
            if p.discount_price is not None else None
        ),
        discount_amount=(
            float(p.discount_amount)
            if p.discount_amount is not None else None
        ),
        availability=p.availability,
        url=p.url or "",
        main_image=p.main_image,
        last_checked=p.last_checked,
        effective_price=effective_price(p.price, p.discount_price),
    )


def _summarize(products: list) -> tuple[
    Optional[str], Optional[float], Optional[float]
]:
    best = None
    worst = None
    for p in products:
        v = effective_price(p.price, p.discount_price)
        if v is None or v <= 0:
            continue
        site = p.site or "unknown"
        if best is None or v < best[0]:
            best = (v, site)
        if worst is None or v > worst[0]:
            worst = (v, site)
    return (
        best[1] if best else None,
        best[0] if best else None,
        worst[0] if worst else None,
    )