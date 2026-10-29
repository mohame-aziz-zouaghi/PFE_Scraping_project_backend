"""
Backfill base_groups + canonical_products from existing products.

Split architecture:
    base_groups          — parent umbrellas (one per model family)
    canonical_products   — variants (one per exact SKU)

Deterministic UUIDs. Split transactions. Batched updates.

Run:
    python -m scripts.backfill_canonical_products
    python -m scripts.backfill_canonical_products --dry-run
    python -m scripts.backfill_canonical_products --limit 5000
    python -m scripts.backfill_canonical_products --resume
    python -m scripts.backfill_canonical_products --only-unlinked
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
from typing import Optional
from uuid import UUID

_THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = _THIS_FILE.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ═══════════════════════════════════════════════════════
# Deterministic UUIDs
# ═══════════════════════════════════════════════════════
_NS_CANONICAL = uuid.uuid5(uuid.NAMESPACE_OID, "canonical")
_NS_BASE = uuid.uuid5(uuid.NAMESPACE_OID, "base")


def _canonical_uuid(identity_hash: str) -> UUID:
    return uuid.uuid5(_NS_CANONICAL, identity_hash)


def _base_uuid(base_sig: str) -> UUID:
    return uuid.uuid5(_NS_BASE, base_sig)


# ═══════════════════════════════════════════════════════
# Signature helpers
# ═══════════════════════════════════════════════════════
_BRAND_ALIASES = {
    "d-link": "dlink",
    "tp-link": "tplink",
    "hewlett packard": "hp",
    "hewlett-packard": "hp",
    "hp inc": "hp",
    "lenovo ideapad": "lenovo",
    "asus rog": "asus",
    "msi gaming": "msi",
    "apple inc": "apple",
    "samsung electronics": "samsung",
    "lg electronics": "lg",
    "western digital": "wd",
    "seagate technology": "seagate",
}
_NO_BRAND = {
    "sans marque", "sans-marque", "generic", "unbranded",
    "no brand", "noname", "no name", "sans fabricant",
    "sans-fabricant", "n/a", "na", "",
}


def _normalize_brand(brand: Optional[str]) -> Optional[str]:
    if not brand:
        return None
    b = brand.strip().lower()
    if b in _NO_BRAND:
        return None
    return _BRAND_ALIASES.get(b, b)


def _extract_signature(product) -> dict:
    from app.core.variant_signature import extract_variant_signature
    try:
        sig = extract_variant_signature(product) or {}
    except Exception:
        sig = {}
    if not isinstance(sig, dict):
        sig = {}
    return sig


def _identity_hash(sig: dict, brand: Optional[str]) -> str:
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
    hard = sig.get("hard_specs") or {}
    for k in sorted(hard.keys()):
        vals = hard.get(k) or []
        if vals:
            parts.append(f"{k}={sorted(str(x) for x in vals)}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _base_signature(
    brand: Optional[str], name: Optional[str], category: Optional[str]
) -> str:
    from app.core.variant_signature import _strip_attributes
    from app.core.product_tokenizer import normalize as _norm
    base_name = _strip_attributes(_norm(name or ""))
    raw = "|".join([_norm(brand or ""), base_name, _norm(category or "")])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _base_name(brand: Optional[str], canonical_name: Optional[str],
               category: Optional[str]) -> str:
    from app.core.variant_signature import _strip_attributes
    from app.core.product_tokenizer import normalize as _norm
    stripped = _strip_attributes(_norm(canonical_name or "")).strip(" /-|,")
    if stripped:
        return stripped[:500]
    fallback = " ".join(
        p for p in (_norm(brand or ""), _norm(category or "")) if p
    ).strip()
    return (fallback or (canonical_name or "Unknown"))[:500]


# ═══════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════
def _pick_name(names: list[str]) -> str:
    candidates = [n for n in names if n and n.strip()]
    if not candidates:
        return "Unknown"
    candidates.sort(key=lambda n: (len(n), n.lower()))
    return candidates[0][:500]


def _pick_mode(values: list[Optional[str]]) -> Optional[str]:
    cleaned = [v for v in values if v]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _collect_colour_variants(sigs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for s in sigs:
        c = s.get("color")
        if c and c not in seen:
            seen.add(c)
            out.append({"color": c})
    return out


def _merge_signatures(sigs: list[dict]) -> dict:
    if not sigs:
        return {}
    keys: set[str] = set()
    for s in sigs:
        keys.update(s.keys())
    merged: dict = {}
    for k in keys:
        if k in ("color", "category"):
            continue
        values = [s.get(k) for s in sigs if s.get(k)]
        if not values:
            continue
        if all(isinstance(v, list) for v in values):
            acc: list[str] = []
            for v in values:
                for item in v:
                    if item not in acc:
                        acc.append(item)
            merged[k] = sorted(acc)
        elif all(isinstance(v, dict) for v in values):
            md: dict = {}
            for v in values:
                for dk, dv in v.items():
                    md.setdefault(dk, dv)
            merged[k] = md
        else:
            merged[k] = Counter(str(v) for v in values).most_common(1)[0][0]
    return merged


# ═══════════════════════════════════════════════════════
# Bulk DB operations
# ═══════════════════════════════════════════════════════
async def _upsert_bases(db, rows: list[dict]) -> int:
    if not rows:
        return 0
    from sqlalchemy import func, text
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.base_group import BaseGroup

    await db.execute(text("SET LOCAL statement_timeout = 60000"))

    BATCH = 500
    written = 0
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
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
        await db.commit()
        written += len(chunk)
        print(f"    base batch {start + len(chunk)}/{len(rows)}")
    return written


async def _upsert_canonicals(db, rows: list[dict]) -> int:
    if not rows:
        return 0
    from sqlalchemy import func, text
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.canonical_product import CanonicalProduct

    await db.execute(text("SET LOCAL statement_timeout = 60000"))

    BATCH = 500
    written = 0
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
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
        await db.commit()
        written += len(chunk)
        print(f"    canonical batch {start + len(chunk)}/{len(rows)}")
    return written


async def _link_products(db, mapping: dict[UUID, UUID]) -> int:
    """
    Link products to canonicals — batched UNNEST pair update.

    1000 rows per UPDATE, each committed independently, so a slow
    run is resumable and never approaches statement_timeout.
    """
    if not mapping:
        print("  ⚠ mapping is empty — nothing to link")
        return 0

    from sqlalchemy import text

    items = list(mapping.items())
    total = len(items)
    BATCH = 1000
    linked = 0

    print(f"  linking {total} products in batches of {BATCH}")

    for start in range(0, total, BATCH):
        chunk = items[start:start + BATCH]
        pids = [str(pid) for pid, _ in chunk]
        cids = [str(cid) for _, cid in chunk]

        result = await db.execute(
            text("""
                UPDATE products AS p
                SET canonical_id = v.cid
                FROM (
                    SELECT
                        CAST(p_arr AS uuid) AS pid,
                        CAST(c_arr AS uuid) AS cid
                    FROM UNNEST(
                        CAST(:pids AS uuid[]),
                        CAST(:cids AS uuid[])
                    ) AS t(p_arr, c_arr)
                ) AS v
                WHERE p.id = v.pid
                  AND (p.canonical_id IS DISTINCT FROM v.cid)
            """),
            {"pids": pids, "cids": cids},
        )
        await db.commit()
        batch_count = result.rowcount or 0
        linked += batch_count
        print(f"    link batch {start + len(chunk)}/{total} → {batch_count} rows")

    print(f"  ✓ linked {linked} products total")
    return linked


async def _denorm_embeddings(db) -> int:
    """
    Batch the embedding denormalization — no giant UPDATE.

    Reads the list of (product_id) needing update from the SAME
    session (single connection), then batched UPDATEs in 1000s.
    """
    from sqlalchemy import text

    # Phase 1: find what needs updating (same session, no extra connection)
    rows = (await db.execute(text(
        """
        SELECT p.id::text
        FROM products p
        JOIN product_embeddings pe ON pe.product_id = p.id
        WHERE p.canonical_id IS NOT NULL
          AND (pe.canonical_id IS DISTINCT FROM p.canonical_id)
        """
    ))).scalars().all()
    await db.commit()   # close the read snapshot

    if not rows:
        print("  embedding denorm: nothing to update")
        return 0

    total = len(rows)
    BATCH = 1000
    updated = 0
    print(f"  embedding denorm: {total} rows in batches of {BATCH}")

    for start in range(0, total, BATCH):
        chunk = rows[start:start + BATCH]
        result = await db.execute(
            text("""
                UPDATE product_embeddings pe
                SET canonical_id = p.canonical_id
                FROM products p
                WHERE pe.product_id = p.id
                  AND p.id = ANY(CAST(:ids AS uuid[]))
                  AND p.canonical_id IS NOT NULL
                  AND (pe.canonical_id IS DISTINCT FROM p.canonical_id)
            """),
            {"ids": chunk},
        )
        await db.commit()
        updated += result.rowcount or 0
        print(f"    embedding batch {start + len(chunk)}/{total}")

    print(f"  ✓ {updated} embeddings denormalized")
    return updated


async def _update_variant_prices(db) -> None:
    from sqlalchemy import text
    await db.execute(text("SET LOCAL statement_timeout = 120000"))

    await db.execute(text(
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
                AND p.price IS NOT NULL AND p.price > 0
                AND COALESCE(NULLIF(p.discount_price, 0), p.price) > 0
          )
        """
    ))
    await db.execute(text(
        """
        UPDATE canonical_products cp
        SET cheapest_price = sub.min_price,
            cheapest_site  = sub.best_site,
            price_updated_at = NOW()
        FROM (
            SELECT p.canonical_id,
                   MIN(COALESCE(NULLIF(p.discount_price, 0), p.price)) AS min_price,
                   (ARRAY_AGG(p.site
                              ORDER BY COALESCE(NULLIF(p.discount_price, 0), p.price) ASC,
                                       p.site ASC))[1] AS best_site
            FROM products p
            WHERE p.is_enabled = TRUE
              AND p.canonical_id IS NOT NULL
              AND p.site IS NOT NULL
              AND p.price IS NOT NULL AND p.price > 0
              AND COALESCE(NULLIF(p.discount_price, 0), p.price) > 0
            GROUP BY p.canonical_id
        ) sub
        WHERE cp.id = sub.canonical_id
        """
    ))
    await db.commit()


async def _update_base_prices(db) -> None:
    from sqlalchemy import text
    await db.execute(text("SET LOCAL statement_timeout = 120000"))

    await db.execute(text(
        """
        UPDATE base_groups bg
        SET cheapest_price = NULL, cheapest_site = NULL,
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
    await db.execute(text(
        """
        UPDATE base_groups bg
        SET cheapest_price = sub.min_price,
            cheapest_site  = sub.best_site,
            price_updated_at = NOW()
        FROM (
            SELECT cp.base_group_id,
                   MIN(cp.cheapest_price) AS min_price,
                   (ARRAY_AGG(cp.cheapest_site
                              ORDER BY cp.cheapest_price ASC,
                                       cp.cheapest_site ASC))[1] AS best_site
            FROM canonical_products cp
            WHERE cp.base_group_id IS NOT NULL
              AND cp.cheapest_price IS NOT NULL
              AND cp.cheapest_site IS NOT NULL
            GROUP BY cp.base_group_id
        ) sub
        WHERE bg.id = sub.base_group_id
        """
    ))
    await db.commit()


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
async def _run(args: argparse.Namespace) -> int:
    from sqlalchemy import select, text
    from app.db import acquire_db, dispose_engine
    from app.db.redis_client import close_redis
    from app.models.product import Product

    print("=" * 70)
    print("BACKFILL — base_groups + canonical_products (batched, no temp tables)")
    print("=" * 70)
    print(f"  Limit:        {args.limit or 'unlimited'}")
    print(f"  Dry run:      {args.dry_run}")
    print(f"  Only unlinked: {args.only_unlinked}")
    print("=" * 70)

    started = time.monotonic()

    try:
        # ── TX 1: schema check + product load ──────────────────────
        async with acquire_db(readonly=True) as db:
            tables = {r[0] for r in await db.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name IN ('base_groups', 'canonical_products')"
            ))}
            if not {"base_groups", "canonical_products"} <= tables:
                print(f"✗ Missing tables. Found: {sorted(tables)}")
                return 1
            print("✓ Schema verified")

            stmt = select(Product).where(Product.is_enabled.is_(True))
            if args.only_unlinked:
                stmt = stmt.where(Product.canonical_id.is_(None))
            if args.limit:
                stmt = stmt.limit(args.limit)
            stmt = stmt.order_by(Product.id)
            products = list((await db.execute(stmt)).scalars().all())
            print(f"Loaded {len(products)} products")

        if not products:
            print("Nothing to do.")
            return 0

        if args.dry_run:
            _report_dry_run(products)
            return 0

        # ── Pure Python — no DB connection held ─────────────────────
        print("Building groups (no DB connection held)…")
        t = time.monotonic()
        groups: dict[tuple, list] = defaultdict(list)
        sigs_by_pid: dict[UUID, dict] = {}
        failed: list[tuple[UUID, str]] = []
        for p in products:
            try:
                sig = _extract_signature(p)
            except Exception as e:
                failed.append((p.id, repr(e)))
                continue
            sigs_by_pid[p.id] = sig
            brand = _normalize_brand(p.brand)
            h = _identity_hash(sig, brand)
            groups[(brand or "__none__", h)].append(p)
        print(f"  {len(groups)} groups in {time.monotonic()-t:.1f}s")
        if failed:
            print(f"  ⚠ {len(failed)} signature failures")

        t = time.monotonic()
        canonical_rows: list[dict] = []
        product_to_canonical: dict[UUID, UUID] = {}
        canonical_meta: dict[UUID, dict] = {}

        for (brand, h), members in groups.items():
            cid = _canonical_uuid(h)
            names = [m.name or "" for m in members]
            for m in members:
                product_to_canonical[m.id] = cid
            sigs = [sigs_by_pid[m.id] for m in members]
            canonical_rows.append({
                "id": cid,
                "canonical_name": _pick_name(names),
                "canonical_brand": _pick_mode([m.brand for m in members]),
                "canonical_category": _pick_mode([m.category for m in members]),
                "identity_signature": _merge_signatures(sigs),
                "variants": _collect_colour_variants(sigs),
                "listing_count": len(members),
                "site_count": len({m.site for m in members if m.site}),
                "confidence": 0.5,
                "base_group_id": None,
            })
            canonical_meta[cid] = {
                "name": canonical_rows[-1]["canonical_name"],
                "brand": canonical_rows[-1]["canonical_brand"],
                "category": canonical_rows[-1]["canonical_category"],
                "members": members,
            }
        print(f"  {len(canonical_rows)} canonicals in {time.monotonic()-t:.1f}s")

        bases: dict[str, list[UUID]] = defaultdict(list)
        for cid, meta in canonical_meta.items():
            bases[_base_signature(meta["brand"], meta["name"], meta["category"])].append(cid)

        base_rows: list[dict] = []
        variant_to_base: dict[UUID, UUID] = {}
        for bsig, cids in bases.items():
            bid = _base_uuid(bsig)
            members: list = []
            for cid in cids:
                members.extend(canonical_meta[cid]["members"])
                variant_to_base[cid] = bid
            rep_name = _pick_name([canonical_meta[c]["name"] for c in cids])
            base_rows.append({
                "id": bid,
                "base_name": _base_name(
                    _pick_mode([m.brand for m in members]),
                    rep_name,
                    _pick_mode([m.category for m in members]),
                ),
                "base_brand": _pick_mode([m.brand for m in members]),
                "base_category": _pick_mode([m.category for m in members]),
                "base_signature": bsig,
                "variant_count": len(cids),
                "site_count": len({m.site for m in members if m.site}),
                "listing_count": len(members),
            })
        for row in canonical_rows:
            row["base_group_id"] = variant_to_base.get(row["id"])
        print(f"  {len(base_rows)} base rows")

        print(f"  product_to_canonical has {len(product_to_canonical)} entries")
        assert len(product_to_canonical) == len(products), (
            f"mapping size {len(product_to_canonical)} != products {len(products)}"
        )

        # ── TX 2: bases ─────────────────────────────────────────────
        async with acquire_db() as db:
            await _upsert_bases(db, base_rows)

        # ── TX 3: canonicals ────────────────────────────────────────
        async with acquire_db() as db:
            await _upsert_canonicals(db, canonical_rows)

        # ── TX 4: link products ─────────────────────────────────────
        async with acquire_db() as db:
            linked = await _link_products(db, product_to_canonical)
        print(f"  ✓ linked {linked} products")

        # ── TX 5: embedding denorm ──────────────────────────────────
        async with acquire_db() as db:
            await _denorm_embeddings(db)

        # ── TX 6: variant prices ────────────────────────────────────
        async with acquire_db() as db:
            await _update_variant_prices(db)

        # ── TX 7: base prices ───────────────────────────────────────
        async with acquire_db() as db:
            await _update_base_prices(db)

        # ── TX 8: verification ──────────────────────────────────────
        async with acquire_db(readonly=True) as db:
            print()
            print("VERIFICATION")
            print("-" * 70)
            checks = {
                "Base groups":              "SELECT COUNT(*) FROM base_groups",
                "Variant canonicals":       "SELECT COUNT(*) FROM canonical_products",
                "Unlinked products":        "SELECT COUNT(*) FROM products WHERE is_enabled = TRUE AND canonical_id IS NULL",
                "Ghost variants":           "SELECT COUNT(*) FROM canonical_products v WHERE v.listing_count > 0 AND NOT EXISTS (SELECT 1 FROM products p WHERE p.canonical_id = v.id)",
                "Variant count mismatches": "SELECT COUNT(*) FROM canonical_products v WHERE v.listing_count <> (SELECT COUNT(*) FROM products p WHERE p.canonical_id = v.id)",
                "Base count mismatches":    "SELECT COUNT(*) FROM base_groups b WHERE b.variant_count <> (SELECT COUNT(*) FROM canonical_products v WHERE v.base_group_id = b.id)",
                "Orphan variants":          "SELECT COUNT(*) FROM canonical_products WHERE base_group_id IS NULL",
                "Canonicals at price 0":    "SELECT COUNT(*) FROM canonical_products WHERE cheapest_price = 0",
                "Bases at price 0":         "SELECT COUNT(*) FROM base_groups WHERE cheapest_price = 0",
            }
            for label, sql in checks.items():
                v = (await db.execute(text(sql))).scalar_one()
                print(f"  {label:<30} {v}")

        print()
        print("=" * 70)
        print(f"Elapsed: {time.monotonic()-started:.1f}s")
        print("=" * 70)
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"\n✗ BACKFILL FAILED: {type(e).__name__}: {e}")
        raise
    finally:
        try:
            await close_redis()
        except Exception:
            pass
        try:
            await dispose_engine()
        except Exception:
            pass


def _report_dry_run(products: list) -> int:
    hashes: Counter = Counter()
    bad = 0
    for p in products:
        try:
            sig = _extract_signature(p)
        except Exception:
            bad += 1
            continue
        brand = _normalize_brand(p.brand)
        hashes[_identity_hash(sig, brand)] += 1
    print(f"  Distinct variant hashes: {len(hashes)}")
    print(f"  Groups with >1 product:  {sum(1 for v in hashes.values() if v > 1)}")
    print(f"  Signature failures:      {bad}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-unlinked", action="store_true")
    args = parser.parse_args()
    if args.resume:
        args.only_unlinked = True
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())