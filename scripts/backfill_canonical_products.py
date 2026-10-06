"""
Backfill base_groups + canonical_products from existing products.

Split architecture:
    base_groups          — parent umbrellas (one per base product)
    canonical_products   — variants (one per exact SKU)

Deterministic UUIDs. Bulk operations. Verifiable.

Run:
    cd backend
    python -m scripts.backfill_canonical_products
    python -m scripts.backfill_canonical_products --dry-run
    python -m scripts.backfill_canonical_products --limit 5000
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

_THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = _THIS_FILE.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ═══════════════════════════════════════════════════════
# Deterministic UUIDs
# ═══════════════════════════════════════════════════════
def _canonical_uuid(identity_hash: str) -> UUID:
    """Deterministic UUID for a variant canonical."""
    return uuid.uuid5(uuid.NAMESPACE_OID, f"canonical:{identity_hash}")


def _base_uuid(base_sig: str) -> UUID:
    """Deterministic UUID for a base group."""
    return uuid.uuid5(uuid.NAMESPACE_OID, f"base:{base_sig}")


# ═══════════════════════════════════════════════════════
# Signature helpers
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


def _identity_signature_hash(sig: dict, brand: Optional[str] = None) -> str:
    """Deterministic hash of identity-defining attributes + brand."""
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
    brand: Optional[str], name: Optional[str], category: Optional[str]
) -> str:
    from app.core.variant_signature import _strip_attributes
    from app.core.product_tokenizer import normalize as _norm

    base_name = _strip_attributes(_norm(name or ""))
    parts = [_norm(brand or ""), base_name, _norm(category or "")]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _build_base_name(
    brand: Optional[str],
    canonical_name: Optional[str],
    category: Optional[str],
) -> str:
    """Derive a base name by stripping variant-defining attributes.

    Uses the same normalizer/stripper as _build_base_signature so the
    base_name is consistent with the grouping key.
    """
    from app.core.variant_signature import _strip_attributes
    from app.core.product_tokenizer import normalize as _norm

    stripped = _strip_attributes(_norm(canonical_name or ""))
    stripped = stripped.strip(" /-|,")
    if stripped:
        return stripped[:500]

    # Fallback: brand + category, else raw canonical
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


def _pick_category(categories: list[Optional[str]]) -> Optional[str]:
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
# Main
# ═══════════════════════════════════════════════════════
async def _run(args: argparse.Namespace) -> int:
    from sqlalchemy import select, text
    from app.core.logging_config import get_logger
    from app.core.variant_signature import extract_variant_signature
    from app.db.redis_client import close_redis
    from app.db.session import async_session_maker, dispose_engine
    from app.models.product import Product

    logger = get_logger(__name__)

    print("=" * 70)
    print("BACKFILL — base_groups + canonical_products (split architecture)")
    print("=" * 70)
    print(f"  Limit:   {args.limit or 'unlimited'}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)

    started = time.monotonic()

    try:
        async with async_session_maker() as db:
            # ── 0. Schema check
            result = await db.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name IN ('base_groups', 'canonical_products')"
            ))
            tables = {row[0] for row in result}
            if "base_groups" not in tables or "canonical_products" not in tables:
                print(f"✗ Missing tables. Found: {sorted(tables)}")
                return 1
            print("✓ Schema verified")

            # ── 1. Load products
            t = time.monotonic()
            stmt = select(Product).where(Product.is_enabled.is_(True))
            if args.limit:
                stmt = stmt.limit(args.limit)
            stmt = stmt.order_by(Product.id)
            products = list((await db.execute(stmt)).scalars().all())
            print(f"Loaded {len(products)} products in "
                  f"{time.monotonic()-t:.1f}s")

            if args.dry_run:
                _report_dry_run(products)
                return 0

            # ── 2. Group by (brand, identity hash)
            t = time.monotonic()
            groups: dict[tuple, list[Product]] = defaultdict(list)
            signatures: dict[UUID, dict] = {}
            for p in products:
                sig = extract_variant_signature(p)
                signatures[p.id] = sig
                brand = _normalize_brand(p.brand)
                h = _identity_signature_hash(sig, brand=brand)
                key = (brand or "__none__", h)
                groups[key].append(p)
            print(f"Computed {len(groups)} variant groups in "
                  f"{time.monotonic()-t:.1f}s")

            # ── 3. Prepare variant rows (canonical_products)
            t = time.monotonic()
            variant_rows: list[dict] = []
            product_to_canonical: dict[UUID, UUID] = {}
            canonical_meta: dict[UUID, dict] = {}
            seen_canonical_ids: set[UUID] = set()

            for (brand, h), members in groups.items():
                cid = _canonical_uuid(h)
                if cid in seen_canonical_ids:
                    logger.warning(
                        "duplicate_canonical_uuid",
                        canonical_id=str(cid),
                        brand=brand,
                        hash_prefix=h[:12],
                    )
                    continue
                seen_canonical_ids.add(cid)

                sigs = [signatures[m.id] for m in members]
                names = [m.name or "" for m in members]
                brands = [m.brand for m in members]
                categories = [m.category for m in members]

                for m in members:
                    product_to_canonical[m.id] = cid

                variants = _collect_variants(sigs)
                identity_sig = _canonical_signature(sigs)

                variant_rows.append({
                    "id": cid,
                    "canonical_name": _pick_canonical_name(names),
                    "canonical_brand": _pick_brand(brands),
                    "canonical_category": _pick_category(categories),
                    "identity_signature": identity_sig,
                    "variants": variants,
                    "listing_count": len(members),
                    "site_count": len({m.site for m in members if m.site}),
                    "confidence": 0.5,
                    "base_group_id": None,
                })

                canonical_meta[cid] = {
                    "name": variant_rows[-1]["canonical_name"],
                    "brand": variant_rows[-1]["canonical_brand"],
                    "category": variant_rows[-1]["canonical_category"],
                    "members": members,
                }

            print(f"Prepared {len(variant_rows)} variant rows in "
                  f"{time.monotonic()-t:.1f}s")

            # ── 4. Build base groups from variant canonicals
            t = time.monotonic()
            bases: dict[str, list[UUID]] = defaultdict(list)
            for cid, meta in canonical_meta.items():
                base_sig = _build_base_signature(
                    meta["brand"], meta["name"], meta["category"]
                )
                bases[base_sig].append(cid)

            base_rows: list[dict] = []
            variant_to_base: dict[UUID, UUID] = {}
            seen_base_ids: set[UUID] = set()

            for base_sig, cids in bases.items():
                base_id = _base_uuid(base_sig)
                if base_id in seen_base_ids:
                    continue
                seen_base_ids.add(base_id)

                # Aggregate across all member products of all variants
                all_members: list[Product] = []
                for cid in cids:
                    all_members.extend(canonical_meta[cid]["members"])

                # Pick the representative canonical name among the children,
                # then strip attributes to get a proper base name.
                child_canonical_names = [
                    canonical_meta[cid]["name"] for cid in cids
                ]
                representative_name = _pick_canonical_name(
                    child_canonical_names
                )
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

            # Attach base_group_id to variant rows before insert
            for row in variant_rows:
                row["base_group_id"] = variant_to_base.get(row["id"])

            print(f"Prepared {len(base_rows)} base rows in "
                  f"{time.monotonic()-t:.1f}s")

            # ── 5. Upsert base groups first (variants reference them)
            t = time.monotonic()
            await _upsert_bases(db, base_rows)
            print(f"Upserted {len(base_rows)} base groups in "
                  f"{time.monotonic()-t:.1f}s")

            # ── 6. Upsert variant canonicals
            t = time.monotonic()
            await _upsert_canonicals(db, variant_rows)
            print(f"Upserted {len(variant_rows)} variants in "
                  f"{time.monotonic()-t:.1f}s")

            # ── 7. Link products
            t = time.monotonic()
            await _link_products(db, product_to_canonical)
            print(f"Linked {len(product_to_canonical)} products in "
                  f"{time.monotonic()-t:.1f}s")

            # ── 8. Denormalize embeddings
            t = time.monotonic()
            await db.execute(text(
                """
                UPDATE product_embeddings pe
                SET canonical_id = p.canonical_id
                FROM products p
                WHERE pe.product_id = p.id
                  AND p.canonical_id IS NOT NULL
                  AND (pe.canonical_id IS DISTINCT FROM p.canonical_id)
                """
            ))
            print(f"Denormalized embeddings in {time.monotonic()-t:.1f}s")

            # ── 9. Cheapest price per variant
            #
            #  EFFECTIVE PRICE RULE:
            #    discount_price is only used when it is a *positive* number.
            #    A 0 or NULL discount_price means "no discount" → use price.
            #    A 0 or NULL regular price means "no price at all" → exclude
            #    the product from the rollup entirely.
            #
            #  STEP 9a first resets rows that no longer have any valid price
            #  so we don't leave stale cheapest_price values behind.
            t = time.monotonic()

            # 9a — reset canonicals that lost their price coverage
            await db.execute(text(
                """
                UPDATE canonical_products cp
                SET cheapest_price = NULL,
                    cheapest_site  = NULL,
                    price_updated_at = NOW()
                WHERE cp.cheapest_price IS NOT NULL
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
            ))

            # 9b — compute the new cheapest per canonical
            await db.execute(text(
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
                      AND COALESCE(
                              NULLIF(p.discount_price, 0), p.price
                          ) > 0
                    GROUP BY p.canonical_id
                ) sub
                WHERE cp.id = sub.canonical_id
                """
            ))
            print(f"Computed variant cheapest in "
                  f"{time.monotonic()-t:.1f}s")

            # ── 10. Roll up cheapest to base groups
            t = time.monotonic()

            # 10a — reset bases that lost their price coverage
            await db.execute(text(
                """
                UPDATE base_groups bg
                SET cheapest_price = NULL,
                    cheapest_site  = NULL,
                    price_updated_at = NOW()
                WHERE bg.cheapest_price IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM canonical_products cp
                      WHERE cp.base_group_id = bg.id
                        AND cp.cheapest_price IS NOT NULL
                        AND cp.cheapest_site IS NOT NULL
                  )
                """
            ))

            # 10b — recompute the cheapest across children variants
            await db.execute(text(
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
                            ORDER BY
                                cp.cheapest_price ASC,
                                cp.cheapest_site ASC
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
            print(f"Rolled up cheapest to bases in "
                  f"{time.monotonic()-t:.1f}s")

            await db.commit()

            # ── 11. Verification
            print()
            print("VERIFICATION")
            print("-" * 70)

            v = await db.execute(text("SELECT COUNT(*) FROM base_groups"))
            print(f"  Base groups:                {v.scalar_one()}")

            v = await db.execute(text(
                "SELECT COUNT(*) FROM canonical_products"
            ))
            print(f"  Variant canonicals:         {v.scalar_one()}")

            v = await db.execute(text(
                "SELECT COUNT(*) FROM products "
                "WHERE is_enabled = TRUE AND canonical_id IS NULL"
            ))
            print(f"  Unlinked products:          {v.scalar_one()}")

            v = await db.execute(text(
                """
                SELECT COUNT(*) FROM canonical_products v
                WHERE v.listing_count > 0
                  AND NOT EXISTS (
                      SELECT 1 FROM products p
                      WHERE p.canonical_id = v.id
                  )
                """
            ))
            print(f"  Ghost variants:             {v.scalar_one()}")

            v = await db.execute(text(
                """
                SELECT COUNT(*) FROM canonical_products v
                WHERE v.listing_count <> (
                    SELECT COUNT(*) FROM products p
                    WHERE p.canonical_id = v.id
                )
                """
            ))
            print(f"  Variant count mismatches:   {v.scalar_one()}")

            v = await db.execute(text(
                """
                SELECT COUNT(*) FROM base_groups b
                WHERE b.variant_count <> (
                    SELECT COUNT(*) FROM canonical_products v
                    WHERE v.base_group_id = b.id
                )
                """
            ))
            print(f"  Base count mismatches:      {v.scalar_one()}")

            v = await db.execute(text(
                """
                SELECT COUNT(*) FROM base_groups b
                WHERE b.variant_count > 0
                  AND b.cheapest_price IS NULL
                """
            ))
            print(f"  Bases without price:        {v.scalar_one()}")

            v = await db.execute(text(
                """
                SELECT COUNT(*) FROM canonical_products v
                WHERE v.base_group_id IS NULL
                """
            ))
            print(f"  Orphan variants:            {v.scalar_one()}")

            # NEW check — zero-price leakage
            v = await db.execute(text(
                """
                SELECT COUNT(*) FROM canonical_products
                WHERE cheapest_price = 0
                """
            ))
            print(f"  Canonicals at price 0:      {v.scalar_one()}")

            v = await db.execute(text(
                """
                SELECT COUNT(*) FROM base_groups
                WHERE cheapest_price = 0
                """
            ))
            print(f"  Bases at price 0:           {v.scalar_one()}")

            elapsed = time.monotonic() - started
            print()
            print("=" * 70)
            print(f"Elapsed: {elapsed:.1f}s")
            print("=" * 70)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        try:
            await close_redis()
        except Exception:
            pass
        try:
            await dispose_engine()
        except Exception:
            pass

    return 0


# ═══════════════════════════════════════════════════════
# Bulk helpers
# ═══════════════════════════════════════════════════════
async def _upsert_bases(db, rows: list[dict]) -> None:
    if not rows:
        return

    from sqlalchemy import func
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.base_group import BaseGroup

    # Deduplicate by id
    by_id = {row["id"]: row for row in rows}
    unique_rows = list(by_id.values())

    if len(unique_rows) != len(rows):
        print(f"  (deduplicated {len(rows) - len(unique_rows)} base rows)")

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
        await db.execute(stmt)


async def _upsert_canonicals(db, rows: list[dict]) -> None:
    if not rows:
        return

    from sqlalchemy import func
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.canonical_product import CanonicalProduct

    by_id = {row["id"]: row for row in rows}
    unique_rows = list(by_id.values())

    if len(unique_rows) != len(rows):
        print(f"  (deduplicated {len(rows) - len(unique_rows)} variant rows)")

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
        await db.execute(stmt)


async def _link_products(db, mapping: dict[UUID, UUID]) -> None:
    if not mapping:
        return

    from sqlalchemy import text

    items = list(mapping.items())
    BATCH = 5000

    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        values_sql = []
        params: dict[str, Any] = {}
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
        await db.execute(sql, params)


def _report_dry_run(products: list) -> None:
    from app.core.variant_signature import extract_variant_signature
    variant_hashes: Counter = Counter()
    for p in products:
        sig = extract_variant_signature(p)
        brand = _normalize_brand(p.brand)
        variant_hashes[_identity_signature_hash(sig, brand=brand)] += 1
    print(f"  Distinct variant hashes: {len(variant_hashes)}")
    print(f"  Groups with >1 product:  "
          f"{sum(1 for v in variant_hashes.values() if v > 1)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())