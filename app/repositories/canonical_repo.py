"""Repository for canonical_products + base_groups.

Single read/write surface for the split schema. The matching service
and the comparison service both use this — no more drift between the
two (Gap 12, Gap 13).
"""
from __future__ import annotations

from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base_group import BaseGroup
from app.models.canonical_product import CanonicalProduct
from app.models.product import Product


class CanonicalRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ═══════════════════════════════════════════════════════
    # Reads
    # ═══════════════════════════════════════════════════════
    async def get(self, canonical_id: UUID) -> Optional[CanonicalProduct]:
        return await self.db.get(CanonicalProduct, canonical_id)

    async def get_base(self, base_id: UUID) -> Optional[BaseGroup]:
        return await self.db.get(BaseGroup, base_id)

    async def get_base_by_signature(
        self, base_signature: str
    ) -> Optional[BaseGroup]:
        stmt = select(BaseGroup).where(
            BaseGroup.base_signature == base_signature
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def get_listings(self, canonical_id: UUID) -> list[Product]:
        stmt = (
            select(Product)
            .where(Product.canonical_id == canonical_id)
            .order_by(Product.site, Product.id)
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def get_variants(
        self, base_id: UUID
    ) -> list[CanonicalProduct]:
        stmt = (
            select(CanonicalProduct)
            .where(CanonicalProduct.base_group_id == base_id)
            .order_by(CanonicalProduct.canonical_name)
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def get_for_product(
        self, product_id: UUID
    ) -> Optional[UUID]:
        row = await self.db.execute(
            select(Product.canonical_id).where(Product.id == product_id)
        )
        return row.scalar_one_or_none()

    # ═══════════════════════════════════════════════════════
    # Writes — variants
    # ═══════════════════════════════════════════════════════
    async def create_variant_group(
        self,
        *,
        canonical_name: str,
        canonical_brand: Optional[str],
        canonical_category: Optional[str],
        identity_signature: dict,
        variants: Optional[list[dict]] = None,
        listing_count: int,
        site_count: int,
        confidence: float,
        base_group_id: Optional[UUID] = None,
    ) -> UUID:
        obj = CanonicalProduct(
            id=uuid4(),
            canonical_name=canonical_name,
            canonical_brand=canonical_brand,
            canonical_category=canonical_category,
            identity_signature=identity_signature or {},
            variants=variants or [],
            listing_count=listing_count,
            site_count=site_count,
            confidence=confidence,
            base_group_id=base_group_id,
        )
        self.db.add(obj)
        await self.db.flush()
        return obj.id

    async def upsert_variants(self, rows: list[dict]) -> None:
        """Bulk upsert used by the backfill script."""
        if not rows:
            return
        by_id = {r["id"]: r for r in rows}
        unique = list(by_id.values())
        BATCH = 500
        for start in range(0, len(unique), BATCH):
            chunk = unique[start:start + BATCH]
            stmt = pg_insert(CanonicalProduct).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "canonical_name": stmt.excluded.canonical_name,
                    "canonical_brand": stmt.excluded.canonical_brand,
                    "canonical_category": stmt.excluded.canonical_category,
                    "identity_signature": stmt.excluded.identity_signature,
                    "variants": stmt.excluded.variants,
                    "listing_count": stmt.excluded.listing_count,
                    "site_count": stmt.excluded.site_count,
                    "confidence": stmt.excluded.confidence,
                    "base_group_id": stmt.excluded.base_group_id,
                    "updated_at": func.now(),
                },
            )
            await self.db.execute(stmt)

    async def assign_products_to_group(
        self, product_ids: list[UUID], canonical_id: UUID
    ) -> None:
        if not product_ids:
            return
        BATCH = 5000
        for start in range(0, len(product_ids), BATCH):
            chunk = product_ids[start:start + BATCH]
            await self.db.execute(
                update(Product)
                .where(Product.id.in_(chunk))
                .values(canonical_id=canonical_id)
            )

    # ═══════════════════════════════════════════════════════
    # Writes — bases
    # ═══════════════════════════════════════════════════════
    async def upsert_base_group(
        self,
        *,
        id: Optional[UUID] = None,
        base_name: str,
        base_brand: Optional[str],
        base_category: Optional[str],
        base_signature: str,
        variant_count: int,
        site_count: int,
        listing_count: int,
    ) -> UUID:
        """Insert or update by base_signature (unique)."""
        existing = await self.get_base_by_signature(base_signature)
        if existing is None:
            obj = BaseGroup(
                id=id or uuid4(),
                base_name=base_name,
                base_brand=base_brand,
                base_category=base_category,
                base_signature=base_signature,
                variant_count=variant_count,
                site_count=site_count,
                listing_count=listing_count,
            )
            self.db.add(obj)
            await self.db.flush()
            return obj.id

        existing.base_name = base_name
        existing.base_brand = base_brand
        existing.base_category = base_category
        existing.variant_count = variant_count
        existing.site_count = site_count
        existing.listing_count = listing_count
        existing.updated_at = func.now()
        await self.db.flush()
        return existing.id

    async def upsert_bases(self, rows: list[dict]) -> None:
        """Bulk upsert used by the backfill script."""
        if not rows:
            return
        by_id = {r["id"]: r for r in rows}
        unique = list(by_id.values())
        BATCH = 500
        for start in range(0, len(unique), BATCH):
            chunk = unique[start:start + BATCH]
            stmt = pg_insert(BaseGroup).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "base_name": stmt.excluded.base_name,
                    "base_brand": stmt.excluded.base_brand,
                    "base_category": stmt.excluded.base_category,
                    "base_signature": stmt.excluded.base_signature,
                    "variant_count": stmt.excluded.variant_count,
                    "site_count": stmt.excluded.site_count,
                    "listing_count": stmt.excluded.listing_count,
                    "updated_at": func.now(),
                },
            )
            await self.db.execute(stmt)

    async def assign_variants_to_base(
        self, variant_ids: list[UUID], base_id: UUID
    ) -> None:
        if not variant_ids:
            return
        await self.db.execute(
            update(CanonicalProduct)
            .where(CanonicalProduct.id.in_(variant_ids))
            .values(base_group_id=base_id, updated_at=func.now())
        )

    # ═══════════════════════════════════════════════════════
    # Maintenance
    # ═══════════════════════════════════════════════════════
    async def clear_all(self) -> None:
        """Truncate derived tables atomically.

        TRUNCATE (not DELETE) so PostgreSQL handles FK cascades without
        triggering the ON DELETE CASCADE on product_match_pairs — which
        would silently destroy the audit trail (Gap 9).

        Order: pairs → variants → bases.
        """
        await self.db.execute(text(
            "TRUNCATE product_match_pairs, canonical_products, base_groups "
            "RESTART IDENTITY CASCADE"
        ))

    async def sync_embedding_canonical_ids(self) -> None:
        """Denormalize products.canonical_id → product_embeddings.canonical_id.

        Idempotent. Cheap. Run after every matching pass (Gap 14).
        """
        await self.db.execute(text(
            """
            UPDATE product_embeddings pe
            SET canonical_id = p.canonical_id
            FROM products p
            WHERE pe.product_id = p.id
              AND p.canonical_id IS NOT NULL
              AND (pe.canonical_id IS DISTINCT FROM p.canonical_id)
            """
        ))

    async def recompute_prices(self) -> None:
        """Recompute cheapest_price rollups for variants and bases.

        Effective price rule: discount_price is used only when it's a
        positive number. 0/NULL means "no discount" and we fall back to
        the regular price. A 0/NULL regular price means "no price at
        all" and the product is excluded from the rollup.
        """
        # Step 1 — reset canonicals that lost price coverage
        await self.db.execute(text(
            """
            UPDATE canonical_products cp
            SET cheapest_price = NULL,
                cheapest_site  = NULL,
                price_updated_at = NOW()
            WHERE cp.cheapest_price IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM products p
                  WHERE p.canonical_id = cp.id
                    AND p.is_enabled = TRUE
                    AND p.price IS NOT NULL
                    AND p.price > 0
                    AND p.site IS NOT NULL
                    AND COALESCE(NULLIF(p.discount_price, 0), p.price) > 0
              )
            """
        ))
        # Step 2 — recompute variants
        await self.db.execute(text(
            """
            UPDATE canonical_products cp
            SET cheapest_price = sub.min_price,
                cheapest_site  = sub.best_site,
                price_updated_at = NOW()
            FROM (
                SELECT
                    p.canonical_id,
                    MIN(COALESCE(NULLIF(p.discount_price, 0), p.price))
                        AS min_price,
                    (ARRAY_AGG(
                        p.site
                        ORDER BY
                            COALESCE(NULLIF(p.discount_price, 0), p.price)
                                ASC,
                            p.site ASC
                    ))[1] AS best_site
                FROM products p
                WHERE p.is_enabled = TRUE
                  AND p.canonical_id IS NOT NULL
                  AND p.site IS NOT NULL
                  AND p.price IS NOT NULL
                  AND p.price > 0
                  AND COALESCE(NULLIF(p.discount_price, 0), p.price) > 0
                GROUP BY p.canonical_id
            ) sub
            WHERE cp.id = sub.canonical_id
            """
        ))
        # Step 3 — reset bases that lost coverage
        await self.db.execute(text(
            """
            UPDATE base_groups bg
            SET cheapest_price = NULL,
                cheapest_site  = NULL,
                price_updated_at = NOW()
            WHERE bg.cheapest_price IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM canonical_products cp
                  WHERE cp.base_group_id = bg.id
                    AND cp.cheapest_price IS NOT NULL
                    AND cp.cheapest_site IS NOT NULL
              )
            """
        ))
        # Step 4 — recompute bases
        await self.db.execute(text(
            """
            UPDATE base_groups bg
            SET cheapest_price = sub.min_price,
                cheapest_site  = sub.best_site,
                price_updated_at = NOW()
            FROM (
                SELECT
                    cp.base_group_id,
                    MIN(cp.cheapest_price) AS min_price,
                    (ARRAY_AGG(
                        cp.cheapest_site
                        ORDER BY cp.cheapest_price ASC, cp.cheapest_site ASC
                    ))[1] AS best_site
                FROM canonical_products cp
                WHERE cp.base_group_id IS NOT NULL
                  AND cp.cheapest_price IS NOT NULL
                  AND cp.cheapest_site IS NOT NULL
                GROUP BY cp.base_group_id
            ) sub
            WHERE bg.id = sub.base_group_id
            """
        ))