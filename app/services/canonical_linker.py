# app/services/canonical_linker.py
"""
Live canonical linkage.

The batch equivalent of `scripts.backfill_canonical_products`,
extracted into a service so it can run from the embedding queue
on every new product.

For a batch of product IDs:

    1. Load the products.
    2. Compute each product's identity signature.
    3. Group the batch by canonical_id.
    4. Upsert base_groups.
    5. Upsert canonical_products.
    6. Link products to their canonical.
    7. Recompute cheapest_price for the affected canonicals.

Idempotent — running twice produces the same rows.

Unlike the backfill script, this does NOT touch
product_embeddings.canonical_id (the embedding pipeline handles
that separately via `embed_products_bulk`). The two steps are
independent, which is what allows them to run in either order.

Safe to run on 1 product or 1000; internally batches SQL at 500
rows to stay under the asyncpg parameter cap.
"""
from __future__ import annotations

import hashlib
import uuid
from collections import Counter, defaultdict
from typing import Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.models.base_group import BaseGroup
from app.models.canonical_product import CanonicalProduct
from app.models.product import Product

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Deterministic UUIDs (identical to the backfill script)
# ═══════════════════════════════════════════════════════
def _canonical_uuid(identity_hash: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_OID, f"canonical:{identity_hash}")


def _base_uuid(base_sig: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_OID, f"base:{base_sig}")


# ═══════════════════════════════════════════════════════
# Signature helpers (identical to the backfill script)
# ═══════════════════════════════════════════════════════
def _normalize_brand(brand: Optional[str]) -> Optional[str]:
    if not brand:
        return None
    b = brand.strip().lower()
    if b in {
        "sans marque", "sans-marque", "generic", "unbranded",
        "no brand", "noname", "no name", "sans fabricant",
        "sans-fabricant",
    }:
        return None
    return {
        "d-link": "dlink",
        "tp-link": "tplink",
        "hewlett packard": "hp",
    }.get(b, b)


def _identity_signature_hash(
    sig: dict, brand: Optional[str] = None
) -> str:
    keys = (
        "model", "capacities", "screen", "size",
        "gpu_models", "cpu_models", "ryzen_models",
    )
    parts = [f"brand={brand or '__none__'}"]
    for k in keys:
        v = sig.get(k)
        if isinstance(v, list):
            v = sorted(str(x) for x in v)
        parts.append(f"{k}={v}")
    hs = sig.get("hard_specs", {}) or {}
    for k in sorted(hs.keys()):
        vals = hs.get(k) or []
        if vals:
            parts.append(f"{k}={sorted(str(x) for x in vals)}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_base_signature(
    brand: Optional[str],
    name: Optional[str],
    category: Optional[str],
) -> str:
    from app.core.variant_signature import _strip_attributes
    from app.core.product_tokenizer import normalize as _norm

    base_name = _strip_attributes(_norm(name or ""))
    parts = [_norm(brand or ""), base_name, _norm(category or "")]
    return hashlib.sha256(
        "|".join(parts).encode("utf-8")
    ).hexdigest()


def _build_base_name(
    brand: Optional[str],
    canonical_name: Optional[str],
    category: Optional[str],
) -> str:
    from app.core.variant_signature import _strip_attributes
    from app.core.product_tokenizer import normalize as _norm

    stripped = _strip_attributes(_norm(canonical_name or ""))
    stripped = stripped.strip(" /-|,")
    if stripped:
        return stripped[:500]

    fallback = " ".join(
        p for p in (_norm(brand or ""), _norm(category or "")) if p
    ).strip()
    return (fallback or (canonical_name or "Unknown"))[:500]


def _pick_canonical_name(names: list[str]) -> str:
    if not names:
        return "Unknown"
    candidates = [n for n in names if n and n.strip()]
    if not candidates:
        return "Unknown"
    candidates.sort(key=lambda n: (len(n), n.lower()))
    return candidates[0][:500]


def _pick_brand(brands: list[Optional[str]]) -> Optional[str]:
    cleaned = [b for b in brands if b]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _pick_category(
    categories: list[Optional[str]]
) -> Optional[str]:
    cleaned = [c for c in categories if c]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _collect_variants(signatures: list[dict]) -> list[dict]:
    seen: set[str] = set()
    variants: list[dict] = []
    for sig in signatures:
        color = sig.get("color")
        if color and color not in seen:
            seen.add(color)
            variants.append({"color": color})
    return variants


def _canonical_signature(sigs: list[dict]) -> dict:
    if not sigs:
        return {}
    keys: set[str] = set()
    for s in sigs:
        keys.update(s.keys())
    result: dict = {}
    for k in keys:
        if k in ("color", "category"):
            continue
        values = [s.get(k) for s in sigs if s.get(k)]
        if not values:
            continue
        if all(isinstance(v, list) for v in values):
            merged: list[str] = []
            for v in values:
                for item in v:
                    if item not in merged:
                        merged.append(item)
            result[k] = sorted(merged)
        elif all(isinstance(v, dict) for v in values):
            merged_dict: dict = {}
            for v in values:
                for dk, dv in v.items():
                    if dk not in merged_dict:
                        merged_dict[dk] = dv
            result[k] = merged_dict
        else:
            result[k] = Counter(
                str(v) for v in values
            ).most_common(1)[0][0]
    return result


# ═══════════════════════════════════════════════════════
# Public service
# ═══════════════════════════════════════════════════════
class CanonicalLinker:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def link_products(
        self, product_ids: list[UUID]
    ) -> dict[str, int]:
        """
        Link a batch of products to their canonical + base
        group, creating those rows if they don't exist.

        Returns a small stats dict for logging.
        """
        if not product_ids:
            return {"linked": 0, "canonicals": 0, "bases": 0}

        from app.core.variant_signature import extract_variant_signature

        # ── 1. Load products
        rows = list((await self.db.execute(
            select(Product).where(
                Product.id.in_(product_ids),
                Product.is_enabled.is_(True),
            )
        )).scalars().all())
        if not rows:
            return {"linked": 0, "canonicals": 0, "bases": 0}

        # ── 2. Compute signatures
        signatures: dict[UUID, dict] = {}
        groups: dict[tuple, list[Product]] = defaultdict(list)
        for p in rows:
            sig = extract_variant_signature(p)
            signatures[p.id] = sig
            brand = _normalize_brand(p.brand)
            h = _identity_signature_hash(sig, brand=brand)
            groups[(brand or "__none__", h)].append(p)

        # ── 3. Prepare canonical rows
        variant_rows: list[dict] = []
        product_to_canonical: dict[UUID, UUID] = {}
        canonical_meta: dict[UUID, dict] = {}

        for (brand, h), members in groups.items():
            cid = _canonical_uuid(h)
            sigs = [signatures[m.id] for m in members]

            for m in members:
                product_to_canonical[m.id] = cid

            variant_rows.append({
                "id": cid,
                "canonical_name": _pick_canonical_name(
                    [m.name or "" for m in members]
                ),
                "canonical_brand": _pick_brand(
                    [m.brand for m in members]
                ),
                "canonical_category": _pick_category(
                    [m.category for m in members]
                ),
                "identity_signature": _canonical_signature(sigs),
                "variants": _collect_variants(sigs),
                "listing_count": len(members),
                "site_count": len(
                    {m.site for m in members if m.site}
                ),
                "confidence": 0.5,
                "base_group_id": None,
            })

            canonical_meta[cid] = {
                "name": variant_rows[-1]["canonical_name"],
                "brand": variant_rows[-1]["canonical_brand"],
                "category": variant_rows[-1]["canonical_category"],
                "members": members,
            }

        # ── 4. Build base groups from the canonicals
        bases: dict[str, list[UUID]] = defaultdict(list)
        for cid, meta in canonical_meta.items():
            base_sig = _build_base_signature(
                meta["brand"], meta["name"], meta["category"]
            )
            bases[base_sig].append(cid)

        base_rows: list[dict] = []
        variant_to_base: dict[UUID, UUID] = {}

        for base_sig, cids in bases.items():
            base_id = _base_uuid(base_sig)

            all_members: list[Product] = []
            for cid in cids:
                all_members.extend(canonical_meta[cid]["members"])

            child_names = [
                canonical_meta[cid]["name"] for cid in cids
            ]
            representative_name = _pick_canonical_name(child_names)
            base_brand = _pick_brand(
                [m.brand for m in all_members]
            )
            base_category = _pick_category(
                [m.category for m in all_members]
            )

            for cid in cids:
                variant_to_base[cid] = base_id

            base_rows.append({
                "id": base_id,
                "base_name": _build_base_name(
                    base_brand, representative_name, base_category
                ),
                "base_brand": base_brand,
                "base_category": base_category,
                "base_signature": base_sig,
                "variant_count": len(cids),
                "site_count": len(
                    {m.site for m in all_members if m.site}
                ),
                "listing_count": len(all_members),
            })

        for row in variant_rows:
            row["base_group_id"] = variant_to_base.get(row["id"])

        # ── 5. Upsert base_groups (variants FK to them)
        await self._upsert_bases(base_rows)

        # ── 6. Upsert canonical_products
        await self._upsert_canonicals(variant_rows)

        # ── 7. Link products
        await self._link_products(product_to_canonical)

        # ── 8. Recompute cheapest_price for affected canonicals
        await self._recompute_cheapest(list(canonical_meta.keys()))

        await self.db.commit()

        logger.info(
            "canonical_link_ok",
            products=len(rows),
            canonicals=len(canonical_meta),
            bases=len(base_rows),
        )
        return {
            "linked": len(product_to_canonical),
            "canonicals": len(canonical_meta),
            "bases": len(base_rows),
        }

    # ═══════════════════════════════════════════════════
    # Upserts
    # ═══════════════════════════════════════════════════
    async def _upsert_bases(self, rows: list[dict]) -> None:
        if not rows:
            return
        from sqlalchemy import func

        by_id = {row["id"]: row for row in rows}
        unique_rows = list(by_id.values())

        BATCH = 500
        for start in range(0, len(unique_rows), BATCH):
            chunk = unique_rows[start:start + BATCH]
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

    async def _upsert_canonicals(self, rows: list[dict]) -> None:
        if not rows:
            return
        from sqlalchemy import func

        by_id = {row["id"]: row for row in rows}
        unique_rows = list(by_id.values())

        BATCH = 500
        for start in range(0, len(unique_rows), BATCH):
            chunk = unique_rows[start:start + BATCH]
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

    async def _link_products(
        self, mapping: dict[UUID, UUID]
    ) -> None:
        if not mapping:
            return
        items = list(mapping.items())
        BATCH = 5000

        for start in range(0, len(items), BATCH):
            chunk = items[start:start + BATCH]
            values_sql = []
            params: dict[str, object] = {}
            for i, (pid, cid) in enumerate(chunk):
                values_sql.append(
                    f"(CAST(:p_{i} AS uuid), CAST(:c_{i} AS uuid))"
                )
                params[f"p_{i}"] = str(pid)
                params[f"c_{i}"] = str(cid)

            sql = text(
                "UPDATE products AS p "
                "SET canonical_id = v.cid "
                "FROM (VALUES " + ", ".join(values_sql) + ") "
                "AS v(pid, cid) "
                "WHERE p.id = v.pid"
            )
            await self.db.execute(sql, params)

    async def _recompute_cheapest(
        self, canonical_ids: list[UUID]
    ) -> None:
        """
        Recompute cheapest_price + cheapest_site for the given
        canonicals. Only touches the affected rows, unlike the
        backfill script which rewrites every canonical.
        """
        if not canonical_ids:
            return

        # 9a — reset canonicals in this batch that lost coverage
        await self.db.execute(text(
            """
            UPDATE canonical_products cp
            SET cheapest_price = NULL,
                cheapest_site  = NULL,
                price_updated_at = NOW()
            WHERE cp.id = ANY(CAST(:ids AS uuid[]))
              AND cp.cheapest_price IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM products p
                  WHERE p.canonical_id = cp.id
                    AND p.is_enabled = TRUE
                    AND p.price IS NOT NULL
                    AND p.price > 0
                    AND COALESCE(
                            NULLIF(p.discount_price, 0), p.price
                        ) > 0
              )
            """
        ), {"ids": [str(x) for x in canonical_ids]})

        # 9b — compute the new cheapest for this batch
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
                WHERE p.canonical_id = ANY(CAST(:ids AS uuid[]))
                  AND p.is_enabled = TRUE
                  AND p.site IS NOT NULL
                  AND p.price IS NOT NULL
                  AND p.price > 0
                  AND COALESCE(
                          NULLIF(p.discount_price, 0), p.price
                      ) > 0
                GROUP BY p.canonical_id
            ) sub
            WHERE cp.id = sub.canonical_id
            """
        ), {"ids": [str(x) for x in canonical_ids]})