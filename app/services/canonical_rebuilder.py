"""Deterministic canonical rebuilder.

Groups products by intrinsic identity attributes — the ones that
are the same regardless of which site listed the product.

Key principle: never include site-specific identifiers in the
hash (`special_id`, verbatim `description`, verbatim `name`).
Two products that are the same SKU on two different sites MUST
hash to the same value.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from typing import Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.core.variant_signature import (
    _strip_attributes,
    extract_variant_signature,
)
from app.core.product_tokenizer import normalize as _norm
from app.models.base_group import BaseGroup
from app.models.canonical_product import CanonicalProduct
from app.models.product import Product
from app.services.base_group_debug_logger import BaseGroupDebugLogger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Deterministic UUIDs
# ═══════════════════════════════════════════════════════
def canonical_uuid(identity_hash: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_OID, f"canonical:{identity_hash}")


def base_uuid(base_sig: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_OID, f"base:{base_sig}")


# ═══════════════════════════════════════════════════════
# Brand normalization
# ═══════════════════════════════════════════════════════
_BRAND_MAP = {
    "d-link": "dlink",
    "tp-link": "tplink",
    "hewlett packard": "hp",
    "hewlett-packard": "hp",
    "samsung electronics": "samsung",
    "lg electronics": "lg",
    "western digital": "wd",
}

_NO_BRAND = {
    "sans marque", "sans-marque", "generic", "unbranded",
    "no brand", "noname", "no name", "sans fabricant",
    "sans-fabricant", "",
}


def normalize_brand(brand: Optional[str]) -> Optional[str]:
    if not brand:
        return None
    b = brand.strip().lower()
    if b in _NO_BRAND:
        return None
    return _BRAND_MAP.get(b, b)


# ═══════════════════════════════════════════════════════
# Intrinsic attribute extraction from description
#
# These are attributes that appear in EVERY site's
# description for the same product, written consistently.
# Extracting them lets us distinguish i3/i5/i7 (different
# SKUs) while still grouping the same SKU across sites.
# ═══════════════════════════════════════════════════════

# Intel: "i3-14100", "Core i5 14500", "Intel Core i7-14700"
_INTEL_CPU_RE = re.compile(
    r"\bi([3579])[-\s]?(\d{4,5}[a-z]{0,2})\b",
    re.IGNORECASE,
)

# AMD Ryzen: "Ryzen 5 5600X", "Ryzen 7 7800X3D"
_AMD_CPU_RE = re.compile(
    r"\bryzen\s+([3579])\s+(\d{4}[a-z0-9]{0,4})\b",
    re.IGNORECASE,
)

# Celeron/Pentium/Athlon: less common but worth capturing
_BUDGET_CPU_RE = re.compile(
    r"\b(celeron|pentium|athlon)\s+([a-z0-9]+)\b",
    re.IGNORECASE,
)

# GPU: "RTX 4060", "GTX 1660", "RX 6600"
_NVIDIA_RE = re.compile(
    r"\b(rtx|gtx)\s?(\d{4})\b", re.IGNORECASE,
)
_AMD_GPU_RE = re.compile(
    r"\brx\s?(\d{4})\b", re.IGNORECASE,
)
_INTEL_GPU_RE = re.compile(
    r"\b(iris\s+xe|intel\s+uhd|uhd\s+graphics|iris\s+graphics)\b",
    re.IGNORECASE,
)


def _extract_cpu_key(text_str: Optional[str]) -> Optional[str]:
    """
    Extract a normalized CPU key from any text. Consistent
    across sites because every site writes the same CPU
    string somewhere in the description.
    """
    if not text_str:
        return None
    s = text_str

    m = _INTEL_CPU_RE.search(s)
    if m:
        return f"intel-i{m.group(1)}-{m.group(2).lower()}"

    m = _AMD_CPU_RE.search(s)
    if m:
        return f"amd-ryzen{m.group(1)}-{m.group(2).lower()}"

    m = _BUDGET_CPU_RE.search(s)
    if m:
        return f"{m.group(1).lower()}-{m.group(2).lower()}"

    return None


def _extract_gpu_key(text_str: Optional[str]) -> Optional[str]:
    """
    Extract a normalized GPU key from any text.
    """
    if not text_str:
        return None
    s = text_str

    m = _NVIDIA_RE.search(s)
    if m:
        return f"{m.group(1).lower()}-{m.group(2)}"

    m = _AMD_GPU_RE.search(s)
    if m:
        return f"rx-{m.group(1)}"

    m = _INTEL_GPU_RE.search(s)
    if m:
        # Normalize all Intel integrated GPUs to one key
        return "intel-integrated"

    return None


# ═══════════════════════════════════════════════════════
# Identity signature hash — lenient + CPU/GPU aware
# ═══════════════════════════════════════════════════════
def identity_signature_hash(
    sig: dict,
    brand: Optional[str] = None,
    *,
    description: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Deterministic hash of INTRINSIC product attributes.

    Groups products that share:
      - brand
      - model
      - capacities, screen, size
      - gpu_models, cpu_models, ryzen_models (from signature)
      - hard_specs
      - CPU extracted from description (fallback)
      - GPU extracted from description (fallback)

    Does NOT use:
      - special_id (varies per site)
      - description verbatim (varies per site)
      - name verbatim (varies per site)

    This is what makes cross-site grouping work: the same SKU
    at two different sites has the same intrinsic attributes
    even though the vendor SKUs, descriptions, and marketing
    names all differ.
    """
    parts: list[str] = [f"brand={brand or '__none__'}"]

    # ── Intrinsic signature attributes
    for k in (
        "model", "capacities", "screen", "size",
        "gpu_models", "cpu_models", "ryzen_models",
    ):
        v = sig.get(k)
        if isinstance(v, list):
            v = sorted(str(x) for x in v)
        parts.append(f"{k}={v}")

    # ── Hard specs
    hs = sig.get("hard_specs", {}) or {}
    for k in sorted(hs.keys()):
        vals = hs.get(k) or []
        if vals:
            parts.append(f"{k}={sorted(str(x) for x in vals)}")

    # ── CPU fallback (if signature didn't capture it)
    # This is the key fix for the Dell QCT1250 case:
    # i3/i5/i7 have different CPU extracted from
    # description → different hashes → separate canonicals.
    if not sig.get("cpu_models"):
        cpu_key = (
            _extract_cpu_key(description)
            or _extract_cpu_key(name)
        )
        parts.append(f"cpu={cpu_key or '__none__'}")

    # ── GPU fallback (if signature didn't capture it)
    if not sig.get("gpu_models"):
        gpu_key = (
            _extract_gpu_key(description)
            or _extract_gpu_key(name)
        )
        parts.append(f"gpu={gpu_key or '__none__'}")

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════
# Base signature (model family — unchanged)
# ═══════════════════════════════════════════════════════
def build_base_signature(
    brand: Optional[str],
    name: Optional[str],
    category: Optional[str],
) -> str:
    base_name = _strip_attributes(_norm(name or ""))
    parts = [
        _norm(brand or ""),
        base_name,
        _norm(category or ""),
    ]
    return hashlib.sha256(
        "|".join(parts).encode("utf-8")
    ).hexdigest()


def build_base_name(
    brand: Optional[str],
    canonical_name: Optional[str],
    category: Optional[str],
) -> str:
    stripped = _strip_attributes(_norm(canonical_name or ""))
    stripped = stripped.strip(" /-|,")
    if stripped:
        return stripped[:500]
    fallback = " ".join(
        p for p in (_norm(brand or ""), _norm(category or "")) if p
    ).strip()
    return (fallback or (canonical_name or "Unknown"))[:500]


# ═══════════════════════════════════════════════════════
# Picker helpers (unchanged)
# ═══════════════════════════════════════════════════════
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
# Service
# ═══════════════════════════════════════════════════════
class CanonicalRebuilder:
    def __init__(self, db: AsyncSession):
        self.db = db
        self._debug = BaseGroupDebugLogger(db)

    async def rebuild_for_products(
        self,
        product_ids: list[UUID],
        *,
        task_id: Optional[str] = None,
    ) -> dict:
        stats = {
            "products": 0,
            "canonicals_created": 0,
            "canonicals_updated": 0,
            "bases_created": 0,
            "bases_updated": 0,
            "bases_deleted": 0,
            "orphans_deleted": 0,
        }

        if not product_ids:
            return stats

        rows = list((await self.db.execute(
            select(Product).where(Product.id.in_(product_ids))
        )).scalars().all())
        stats["products"] = len(rows)
        if not rows:
            return stats

        base_changes: list[dict] = []

        # ── 1. Compute signatures + group
        signatures: dict[UUID, dict] = {}
        groups: dict[tuple, list[Product]] = defaultdict(list)
        for p in rows:
            sig = extract_variant_signature(p)
            signatures[p.id] = sig
            brand = normalize_brand(p.brand)
            h = identity_signature_hash(
                sig,
                brand=brand,
                description=p.description,
                name=p.name,
            )
            key = (brand or "__none__", h)
            groups[key].append(p)

        # ── 2. Build canonical rows
        variant_rows: list[dict] = []
        product_to_canonical: dict[UUID, UUID] = {}
        canonical_meta: dict[UUID, dict] = {}

        for (brand, h), members in groups.items():
            cid = canonical_uuid(h)
            sigs = [signatures[m.id] for m in members]
            names = [m.name or "" for m in members]
            brands = [m.brand for m in members]
            categories = [m.category for m in members]

            for m in members:
                product_to_canonical[m.id] = cid

            variant_rows.append({
                "id": cid,
                "canonical_name": _pick_canonical_name(names),
                "canonical_brand": _pick_brand(brands),
                "canonical_category": _pick_category(categories),
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

        # ── 3. Build base groups
        bases: dict[str, list[UUID]] = defaultdict(list)
        for cid, meta in canonical_meta.items():
            base_sig = build_base_signature(
                meta["brand"], meta["name"], meta["category"]
            )
            bases[base_sig].append(cid)

        base_rows: list[dict] = []
        variant_to_base: dict[UUID, UUID] = {}

        for base_sig, cids in bases.items():
            base_id = base_uuid(base_sig)
            all_members: list[Product] = []
            for cid in cids:
                all_members.extend(canonical_meta[cid]["members"])

            representative_name = _pick_canonical_name(
                [canonical_meta[cid]["name"] for cid in cids]
            )
            base_brand = _pick_brand([m.brand for m in all_members])
            base_category = _pick_category(
                [m.category for m in all_members]
            )

            for cid in cids:
                variant_to_base[cid] = base_id

            base_rows.append({
                "id": base_id,
                "base_name": build_base_name(
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

        # ── 4. New vs existing
        existing_base_ids = set((await self.db.execute(
            select(BaseGroup.id).where(
                BaseGroup.id.in_([b["id"] for b in base_rows])
            )
        )).scalars().all())

        for b in base_rows:
            is_new = b["id"] not in existing_base_ids
            if is_new:
                stats["bases_created"] += 1
                change_type = "created"
            else:
                stats["bases_updated"] += 1
                change_type = "updated"

            member_pids: list[UUID] = []
            for cid in bases[b["base_signature"]]:
                member_pids.extend(
                    [m.id for m in canonical_meta[cid]["members"]]
                )

            base_changes.append({
                "base_group_id": b["id"],
                "canonical_id": (
                    bases[b["base_signature"]][0]
                    if len(bases[b["base_signature"]]) == 1
                    else None
                ),
                "change_type": change_type,
                "product_ids": member_pids,
                "details": {
                    "variant_count": b["variant_count"],
                    "site_count": b["site_count"],
                    "listing_count": b["listing_count"],
                },
            })

        # ── 5. Upsert
        await self._upsert_bases(base_rows)

        existing_canonical_ids = set((await self.db.execute(
            select(CanonicalProduct.id).where(
                CanonicalProduct.id.in_(
                    [v["id"] for v in variant_rows]
                )
            )
        )).scalars().all())

        for v in variant_rows:
            if v["id"] in existing_canonical_ids:
                stats["canonicals_updated"] += 1
            else:
                stats["canonicals_created"] += 1

        await self._upsert_canonicals(variant_rows)
        await self._link_products(product_to_canonical)

        # ── 6. Delete orphans
        orphan_canonicals = (await self.db.execute(
            text("""
                DELETE FROM canonical_products cp
                WHERE cp.id = ANY(CAST(:ids AS uuid[]))
                  AND NOT EXISTS (
                      SELECT 1 FROM products p
                      WHERE p.canonical_id = cp.id
                  )
                RETURNING id
            """),
            {"ids": [str(v["id"]) for v in variant_rows]},
        )).all()
        stats["orphans_deleted"] = len(orphan_canonicals)

        orphan_bases_rows = (await self.db.execute(
            text("""
                DELETE FROM base_groups bg
                WHERE bg.id = ANY(CAST(:ids AS uuid[]))
                  AND NOT EXISTS (
                      SELECT 1 FROM canonical_products cp
                      WHERE cp.base_group_id = bg.id
                  )
                RETURNING id
            """),
            {"ids": [str(b["id"]) for b in base_rows]},
        )).all()
        stats["orphans_deleted"] += len(orphan_bases_rows)
        stats["bases_deleted"] = len(orphan_bases_rows)

        for (deleted_bid,) in orphan_bases_rows:
            base_changes.append({
                "base_group_id": deleted_bid,
                "canonical_id": None,
                "change_type": "deleted",
                "product_ids": [],
                "details": {"reason": "no canonicals remain"},
            })

        # ── 7. Recompute cheapest
        from app.services.matching_helpers import (
            recompute_base_cheapest,
            recompute_canonical_cheapest,
        )
        await recompute_canonical_cheapest(
            self.db,
            canonical_ids=[v["id"] for v in variant_rows],
        )
        await recompute_base_cheapest(self.db)

        await self.db.commit()

        await self._debug.log_batch(
            changes=base_changes,
            trigger_source="targeted_rebuild",
            task_id=task_id,
        )
        await self.db.commit()

        logger.info("canonical_rebuild_complete", **stats)
        return stats

    # ═══════════════════════════════════════════════════
    # Upserts (unchanged)
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
        self, mapping: dict[UUID, UUID],
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