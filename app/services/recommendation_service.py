# app/services/recommendation_service.py
"""Top-level recommendation orchestration with priority bands."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.core.recommendation_config import (
    MATERIALIZATION,
    RANKING,
    TIER,
)
from app.models.product import Product
from app.repositories.profile_repo import ProfileRepo
from app.repositories.recommendation_repo import RecommendationRepo
from app.schemas.recommendation import (
    RecommendedItem,
    RecommendationsResponse,
    SimilarProductsResponse,
)
from app.services.candidate_service import CandidateService
from app.services.llm_rerank_service import LLMRerankService
from app.services.priority_bands import PriorityBandService
from app.services.profile_service import ProfileService
from app.services.ranking_service import RankingService, ScoredCandidate

logger = get_logger(__name__)


class RecommendationService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.profiles = ProfileService(db)
        self.profile_repo = ProfileRepo(db)
        self.candidates = CandidateService(db)
        self.ranker = RankingService(db)
        self.reranker = LLMRerankService(db)
        self.repo = RecommendationRepo(db)
        self.bands = PriorityBandService(db)

    # ═══════════════════════════════════════════════════════
    # /for-me
    # ═══════════════════════════════════════════════════════
    async def for_user(
        self, *, user_id: UUID, limit: int = RANKING.DEFAULT_LIMIT,
    ) -> RecommendationsResponse:
        if not settings.recs_enabled:
            return RecommendationsResponse(
                user_id=user_id, cold_start=False, cached=False,
                count=0, items=[],
            )

        limit = min(limit, RANKING.MAX_LIMIT)

        cached_rows = await self.repo.get_fresh_for_user(
            user_id, limit=limit
        )
        if cached_rows:
            products = await self.repo.products_by_ids(
                [r.product_id for r in cached_rows]
            )
            return RecommendationsResponse(
                user_id=user_id,
                cold_start=False,
                cached=True,
                count=len(cached_rows),
                items=self._materialize(cached_rows, products),
            )

        return await self._compute_and_return(
            user_id=user_id, limit=limit
        )

    # ═══════════════════════════════════════════════════════
    # /similar/{product_id}
    # ═══════════════════════════════════════════════════════
    async def similar_to_product(
        self,
        *,
        product_id: UUID,
        limit: int = 8,
    ) -> SimilarProductsResponse:
        if not settings.recs_enabled:
            return SimilarProductsResponse(
                query_product_id=product_id, count=0, items=[]
            )

        product = await self.db.get(Product, product_id)
        if product is None:
            return SimilarProductsResponse(
                query_product_id=product_id, count=0, items=[]
            )

        candidate_ids = await self.candidates._content_neighbours(
            product_id=product_id, limit=max(30, limit * 3)
        )
        if not candidate_ids:
            return SimilarProductsResponse(
                query_product_id=product_id, count=0, items=[]
            )

        scored = await self.ranker.rank(
            candidates=candidate_ids,
            profile=None,
            seen_products={product_id},
            priority_ids=None,
            top_k=max(30, limit * 3),
        )

        products = await self.repo.products_by_ids(
            [s.product_id for s in scored]
        )

        items: list[RecommendedItem] = []
        for s in scored:
            if len(items) >= limit:
                break
            p = products.get(s.product_id)
            if p is None:
                continue
            items.append(self._item_from_scored(s, p))

        return SimilarProductsResponse(
            query_product_id=product_id,
            count=len(items),
            items=items,
        )

    # ═══════════════════════════════════════════════════════
    # Shared computation
    # ═══════════════════════════════════════════════════════
    async def _compute_and_return(
        self, *, user_id: UUID, limit: int
    ) -> RecommendationsResponse:
        # ── Fresh-profile check: rebuild only if expired or missing.
        profile = await self.profile_repo.get_fresh(user_id)
        if profile is None:
            try:
                await self.profiles.rebuild(user_id)
            except Exception as exc:
                await self.db.rollback()
                logger.warning(
                    "profile_rebuild_failed",
                    user_id=str(user_id),
                    error=str(exc)[:200],
                )
                try:
                    await self.db.rollback()
                except Exception:
                    pass
                return RecommendationsResponse(
                    user_id=user_id,
                    cold_start=True,
                    cached=False,
                    count=0,
                    items=[],
                )
            else:
                profile = await self.profile_repo.get_fresh(user_id)

        cold_start = profile is None or profile.is_cold_start

        recent_viewed = await self._recent_seen_products(user_id)

        # ── Priority bands (guaranteed slots).
        wishlist_ids = await self.bands.wishlist_band(
            user_id=user_id, hours=TIER.TODAY_HOURS
        )
        search_ids = await self.bands.search_band(
            user_id=user_id, hours=TIER.TODAY_HOURS
        )
        cart_ids = await self.bands.cart_band(
            user_id=user_id, hours=TIER.TODAY_HOURS
        )
        click_ids = await self.bands.click_band(
            user_id=user_id, hours=TIER.FRESH_HOURS
        )
        dwell_ids = await self.bands.dwell_band(
            user_id=user_id, hours=TIER.FRESH_HOURS
        )

        # Fresh high-intent items must be IN the candidate pool.
        fresh_include = list(dict.fromkeys(
            wishlist_ids + cart_ids + search_ids
        ))

        candidate_ids = await self.candidates.generate(
            user_id=user_id,
            profile=profile,
            exclude=recent_viewed if not cold_start else set(),
            include=fresh_include,
        )

        # ── Priority id sets fed into the general ranker so the
        #    wishlist/search/cart products also get a score boost
        #    if they overflow into the general pool.
        priority_ids = {
            "wishlist": set(wishlist_ids),
            "search":   set(search_ids),
            "cart":     set(cart_ids),
            "click":    set(click_ids),
            "dwell":    set(dwell_ids),
        }

        scored = await self.ranker.rank(
            candidates=candidate_ids,
            profile=profile,
            seen_products=recent_viewed,
            priority_ids=priority_ids,
            top_k=RANKING.STAGE2_TOP_K,
        )

        products = await self.repo.products_by_ids(
            [s.product_id for s in scored] + fresh_include
        )

        # ═══════════════════════════════════════════════════
        # SLOT BLENDING with priority bands.
        # ═══════════════════════════════════════════════════
        selected = self._blend_slots_with_bands(
            scored=scored,
            products=products,
            profile=profile,
            wishlist_ids=wishlist_ids,
            search_ids=search_ids,
            cart_ids=cart_ids,
            click_ids=click_ids,
            dwell_ids=dwell_ids,
        )

        # Safety: drop anything the user just interacted with,
        # except the fresh wishlist/cart items we intentionally
        # want back (they were already filtered out of
        # recent_viewed by _recent_seen_products).
        selected = [
            s for s in selected if s.product_id not in recent_viewed
        ]

        # ═══════════════════════════════════════════════════
        # Stage 3 — LLM rerank (optional)
        # ═══════════════════════════════════════════════════
        reranked = None
        if (
            settings.recs_llm_enabled
            and not cold_start
            and len(selected) >= 5
        ):
            reranked_ids = await self.reranker.rerank(
                user_id=user_id,
                profile=profile,
                candidates=selected,
                products=products,
                recent_view_names=[
                    products[pid].name
                    for pid in list(recent_viewed)[:5]
                    if pid in products and products[pid].name
                ],
                final_limit=limit,
            )
            reranked = reranked_ids

        items = self._build_response_items(
            scored=selected,
            products=products,
            reranked=reranked,
            limit=limit,
        )

        # ── Persist the fresh set for next time.
        try:
            await self.repo.save_batch(
                user_id=user_id,
                rows=[
                    {
                        "product_id": i.product_id,
                        "rank": i.rank,
                        "stage2_score": next(
                            (
                                s.score for s in selected
                                if s.product_id == i.product_id
                            ),
                            0.0,
                        ),
                        "stage3_score": None,
                        "source": i.source,
                        "reason": i.reason,
                    }
                    for i in items
                ],
            )
            await self.db.commit()
        except Exception as exc:
            await self.db.rollback()
            logger.warning(
                "recommendations_persist_failed",
                user_id=str(user_id),
                error=str(exc)[:200],
            )

        return RecommendationsResponse(
            user_id=user_id,
            cold_start=cold_start,
            cached=False,
            count=len(items),
            items=items,
        )

    # ═══════════════════════════════════════════════════════
    # Slot blending with priority bands
    # ═══════════════════════════════════════════════════════
    def _blend_slots_with_bands(
        self,
        *,
        scored: list[ScoredCandidate],
        products: dict[UUID, Product],
        profile,
        wishlist_ids: list[UUID],
        search_ids: list[UUID],
        cart_ids: list[UUID],
        click_ids: list[UUID],
        dwell_ids: list[UUID],
    ) -> list[ScoredCandidate]:
        """
        Assemble the final feed in GUARANTEED bands, in this order:

            1. Wishlist (fresh)     ─ up to BAND_WISHLIST_FRESH slots
            2. Search-match (fresh) ─ up to BAND_SEARCH_FRESH slots
            3. Add-to-cart (fresh)  ─ up to BAND_CART_FRESH slots
            4. Click (fresh)        ─ up to BAND_CLICK_FRESH slots
            5. Dwell (fresh)        ─ up to BAND_DWELL_FRESH slots
            6. General ranker       ─ fills the remaining slots

        If a band is short, unused slots roll down to the next
        band. The feed is filled to TOTAL_SLOTS when the pool is
        large enough.
        """
        by_id: dict[UUID, ScoredCandidate] = {
            s.product_id: s for s in scored
        }

        def _ensure(cid: UUID) -> Optional[ScoredCandidate]:
            """Return a ScoredCandidate for cid, even if the general
            ranker didn't produce one."""
            if cid in by_id:
                return by_id[cid]
            p = products.get(cid)
            if p is None:
                return None
            placeholder = ScoredCandidate(
                product_id=cid,
                score=0.5,
                dominant_signal="wishlist",
                content_sim=0.0,
                price_affinity=0.0,
                brand_affinity=0.0,
                category_affinity=0.0,
                popularity=0.0,
                recency=0.0,
                seen_penalty=0.0,
            )
            by_id[cid] = placeholder
            return placeholder

        selected: list[ScoredCandidate] = []
        chosen_ids: set[UUID] = set()

        def take(
            pool: list[UUID], slots: int, signal: str
        ) -> int:
            """Append up to `slots` items from `pool`; return how
            many were actually taken."""
            taken = 0
            for cid in pool:
                if taken >= slots:
                    break
                if cid in chosen_ids:
                    continue
                s = _ensure(cid)
                if s is None:
                    continue
                s.dominant_signal = signal
                selected.append(s)
                chosen_ids.add(cid)
                taken += 1
            return taken

        # Band 1: wishlist. Unused slots roll to Band 2.
        used_wishlist = take(
            wishlist_ids, TIER.BAND_WISHLIST_FRESH, "wishlist"
        )
        extra_from_wishlist = (
            TIER.BAND_WISHLIST_FRESH - used_wishlist
        )

        # Band 2: search (+ whatever wishlist couldn't fill).
        search_slots = TIER.BAND_SEARCH_FRESH + max(
            0, extra_from_wishlist
        )
        used_search = take(search_ids, search_slots, "search")
        extra_from_search = max(0, search_slots - used_search)

        # Band 3: cart (+ whatever search couldn't fill).
        cart_slots = TIER.BAND_CART_FRESH + extra_from_search
        used_cart = take(cart_ids, cart_slots, "cart")
        extra_from_cart = max(0, cart_slots - used_cart)

        # Band 4: click (+ whatever cart couldn't fill).
        click_slots = TIER.BAND_CLICK_FRESH + extra_from_cart
        used_click = take(click_ids, click_slots, "click")
        extra_from_click = max(0, click_slots - used_click)

        # Band 5: dwell (+ whatever click couldn't fill).
        dwell_slots = TIER.BAND_DWELL_FRESH + extra_from_click
        take(dwell_ids, dwell_slots, "dwell")

        # Band 6: general ranker fills the rest.
        for s in scored:
            if len(selected) >= TIER.TOTAL_SLOTS:
                break
            if s.product_id in chosen_ids:
                continue
            selected.append(s)
            chosen_ids.add(s.product_id)

        return selected[: TIER.TOTAL_SLOTS]

    # ═══════════════════════════════════════════════════════
    # Builders
    # ═══════════════════════════════════════════════════════
    def _build_response_items(
        self,
        *,
        scored,
        products: dict[UUID, Product],
        reranked,
        limit: int,
    ) -> list[RecommendedItem]:
        if reranked is None:
            items: list[RecommendedItem] = []
            for s in scored:
                if len(items) >= limit:
                    break
                p = products.get(s.product_id)
                if p is None:
                    continue
                items.append(self._item_from_scored(s, p))
            for i, item in enumerate(items, start=1):
                item.rank = i
            return items

        by_id = {s.product_id: s for s in scored}
        items: list[RecommendedItem] = []
        for r in reranked:
            s = by_id.get(r.product_id)
            p = products.get(r.product_id)
            if s is None or p is None:
                continue
            item = self._item_from_scored(s, p)
            items.append(
                item.model_copy(update={"reason": r.reason})
            )
            if len(items) >= limit:
                break
        for i, item in enumerate(items, start=1):
            item.rank = i
        return items

    @staticmethod
    def _item_from_scored(s, p: Product) -> RecommendedItem:
        return RecommendedItem(
            product_id=p.id,
            name=p.name or "Untitled",
            brand=p.brand,
            category=p.category,
            price=(
                float(p.price) if p.price is not None else None
            ),
            discount_price=(
                float(p.discount_price)
                if p.discount_price is not None else None
            ),
            availability=p.availability,
            main_image=p.main_image,
            url=p.url,
            site=p.site,
            rank=0,
            score=round(s.score, 4),
            source=s.dominant_signal,
            reason=_template_reason(s, p),
        )

    def _materialize(
        self, rows, products: dict[UUID, Product]
    ) -> list[RecommendedItem]:
        items: list[RecommendedItem] = []
        for r in rows:
            p = products.get(r.product_id)
            if p is None:
                continue
            items.append(RecommendedItem(
                product_id=p.id,
                name=p.name or "Untitled",
                brand=p.brand,
                category=p.category,
                price=(
                    float(p.price) if p.price is not None else None
                ),
                discount_price=(
                    float(p.discount_price)
                    if p.discount_price is not None else None
                ),
                availability=p.availability,
                main_image=p.main_image,
                url=p.url,
                site=p.site,
                rank=r.rank,
                score=float(r.stage2_score),
                source=r.source,
                reason=r.reason,
            ))
        for i, item in enumerate(items, start=1):
            item.rank = i
        return items

    async def _recent_seen_products(
        self, user_id: UUID
    ) -> set[UUID]:
        """
        Products the user recently interacted with, EXCLUDING the
        last FRESH_HOURS of wishlist / cart actions. We want those
        products back in the feed — they're the whole point of the
        priority bands.
        """
        from app.repositories.event_repo import EventRepo

        repo = EventRepo(self.db)
        events = await repo.recent_for_user(
            user_id=user_id, limit=200
        )

        fresh_cutoff = datetime.now(timezone.utc) - timedelta(
            hours=TIER.FRESH_HOURS
        )
        skip_types = {"wishlist", "add_to_cart"}

        out: set[UUID] = set()
        for e in events:
            if e.product_id is None:
                continue
            if (
                e.event_type in skip_types
                and e.created_at >= fresh_cutoff
            ):
                continue
            out.add(e.product_id)
        return out


def _template_reason(s: ScoredCandidate, p: Product) -> str:
    sig = s.dominant_signal
    if sig == "wishlist":
        return "You wishlisted this"
    if sig == "cart":
        return "You added this to cart"
    if sig == "search":
        return "Matches your recent search"
    if sig == "click":
        return "You clicked this recently"
    if sig == "dwell":
        return "You spent time on this"
    if getattr(s, "search_boost", 0.0) > 0:
        return "Matches your recent search"
    if sig == "content":
        return "Similar to products you viewed"
    if sig == "brand":
        return (
            f"More from {p.brand}" if p.brand else "Matches your brands"
        )
    if sig == "category":
        return (
            f"Popular in {p.category}"
            if p.category else "Popular in categories you browse"
        )
    if sig == "price":
        return "Fits your typical budget"
    return "Recommended for you"