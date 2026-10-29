# app/services/matching_helpers.py
"""
Shared primitives for the cross-site canonicalisation layer.

Every piece of SQL that touches canonical_products / base_groups /
product_match_pairs goes through this module. That guarantees:

  - One definition of the "effective price" formula.
  - One definition of the "recompute cheapest" procedure.
  - One definition of the "snapshot tables" procedure.
  - One definition of the "advisory lock" behaviour.

Both app/api/v1/compare.py and app/services/matching_service.py and
the nightly Celery task call into here. If the formula ever changes,
it changes in exactly one place.
"""
from __future__ import annotations

import hmac
from datetime import date
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# Advisory lock key for the matching pass. Any code that wants to
# ensure only one matching run is active must use this exact key.
ADVISORY_LOCK_KEY = 987654321

# Tables the destructive path truncates. Whitelisted here so
# f-string interpolation is provably safe.
SNAPSHOT_TABLES = (
    "canonical_products",
    "base_groups",
    "product_match_pairs",
)


# ═══════════════════════════════════════════════════════
# Effective price
# ═══════════════════════════════════════════════════════
EFFECTIVE_PRICE_SQL = (
    "COALESCE(NULLIF(p.discount_price, 0), p.price)"
)


def effective_price(price, discount_price) -> Optional[float]:
    """Python-side mirror of EFFECTIVE_PRICE_SQL. Used by services that
    already have the row loaded and just need the number."""
    if price is None:
        return None
    if discount_price is not None:
        try:
            dp = float(discount_price)
        except (TypeError, ValueError):
            dp = 0.0
        if dp > 0:
            return dp
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None


# ═══════════════════════════════════════════════════════
# Cheapest recompute
# ═══════════════════════════════════════════════════════
async def recompute_canonical_cheapest(
    db: AsyncSession,
    *,
    canonical_ids: Optional[list] = None,
) -> int:
    """
    Recompute canonical_products.cheapest_price / cheapest_site for
    the given canonical IDs (or all of them if None).

    Idempotent. Resets rows that no longer have any valid price, then
    writes the fresh min. Uses the effective price formula from the
    module constant, so it agrees with every other query in the
    codebase.

    Returns the number of rows updated.
    """
    # Step 1: reset stale prices first (rows that no longer have any
    # valid price).
    if canonical_ids is not None:
        await db.execute(
            text(
                f"""
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
                        AND {EFFECTIVE_PRICE_SQL} > 0
                  )
                """
            ),
            {"ids": [str(x) for x in canonical_ids]},
        )
    else:
        await db.execute(text(
            f"""
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
                    AND {EFFECTIVE_PRICE_SQL} > 0
              )
            """
        ))

    # Step 2: compute the min.
    if canonical_ids is not None:
        result = await db.execute(
            text(
                f"""
                UPDATE canonical_products cp
                SET cheapest_price = sub.min_price,
                    cheapest_site  = sub.best_site,
                    price_updated_at = NOW()
                FROM (
                    SELECT
                        p.canonical_id,
                        MIN({EFFECTIVE_PRICE_SQL}) AS min_price,
                        (ARRAY_AGG(
                            p.site
                            ORDER BY {EFFECTIVE_PRICE_SQL} ASC,
                                     p.site ASC
                        ))[1] AS best_site
                    FROM products p
                    WHERE p.canonical_id = ANY(CAST(:ids AS uuid[]))
                      AND p.is_enabled = TRUE
                      AND p.site IS NOT NULL
                      AND p.price IS NOT NULL
                      AND p.price > 0
                      AND {EFFECTIVE_PRICE_SQL} > 0
                    GROUP BY p.canonical_id
                ) sub
                WHERE cp.id = sub.canonical_id
                """
            ),
            {"ids": [str(x) for x in canonical_ids]},
        )
    else:
        result = await db.execute(text(
            f"""
            UPDATE canonical_products cp
            SET cheapest_price = sub.min_price,
                cheapest_site  = sub.best_site,
                price_updated_at = NOW()
            FROM (
                SELECT
                    p.canonical_id,
                    MIN({EFFECTIVE_PRICE_SQL}) AS min_price,
                    (ARRAY_AGG(
                        p.site
                        ORDER BY {EFFECTIVE_PRICE_SQL} ASC,
                                 p.site ASC
                    ))[1] AS best_site
                FROM products p
                WHERE p.canonical_id IS NOT NULL
                  AND p.is_enabled = TRUE
                  AND p.site IS NOT NULL
                  AND p.price IS NOT NULL
                  AND p.price > 0
                  AND {EFFECTIVE_PRICE_SQL} > 0
                GROUP BY p.canonical_id
            ) sub
            WHERE cp.id = sub.canonical_id
            """
        ))

    return result.rowcount or 0


async def recompute_base_cheapest(
    db: AsyncSession,
    *,
    base_ids: Optional[list] = None,
) -> int:
    """
    Roll the cheapest price up from canonical_products to base_groups.
    Same two-phase reset+write pattern.
    """
    if base_ids is not None:
        await db.execute(
            text(
                """
                UPDATE base_groups bg
                SET cheapest_price = NULL,
                    cheapest_site  = NULL,
                    price_updated_at = NOW()
                WHERE bg.id = ANY(CAST(:ids AS uuid[]))
                  AND bg.cheapest_price IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM canonical_products cp
                      WHERE cp.base_group_id = bg.id
                        AND cp.cheapest_price IS NOT NULL
                        AND cp.cheapest_site  IS NOT NULL
                  )
                """
            ),
            {"ids": [str(x) for x in base_ids]},
        )
        result = await db.execute(
            text(
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
                            ORDER BY cp.cheapest_price ASC,
                                     cp.cheapest_site ASC
                        ))[1] AS best_site
                    FROM canonical_products cp
                    WHERE cp.base_group_id = ANY(CAST(:ids AS uuid[]))
                      AND cp.cheapest_price IS NOT NULL
                      AND cp.cheapest_site  IS NOT NULL
                    GROUP BY cp.base_group_id
                ) sub
                WHERE bg.id = sub.base_group_id
                """
            ),
            {"ids": [str(x) for x in base_ids]},
        )
    else:
        await db.execute(text(
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
                    AND cp.cheapest_site  IS NOT NULL
              )
            """
        ))
        result = await db.execute(text(
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
                        ORDER BY cp.cheapest_price ASC,
                                 cp.cheapest_site ASC
                    ))[1] AS best_site
                FROM canonical_products cp
                WHERE cp.base_group_id IS NOT NULL
                  AND cp.cheapest_price IS NOT NULL
                  AND cp.cheapest_site  IS NOT NULL
                GROUP BY cp.base_group_id
            ) sub
            WHERE bg.id = sub.base_group_id
            """
        ))

    return result.rowcount or 0


# ═══════════════════════════════════════════════════════
# Snapshot
# ═══════════════════════════════════════════════════════
async def snapshot_derived_tables(db: AsyncSession) -> str:
    """
    Copy the three derived tables to date-stamped backups.

    Idempotent (CREATE TABLE IF NOT EXISTS). Returns the date suffix
    so the caller can log it.
    """
    suffix = date.today().strftime("%Y%m%d")
    for tbl in SNAPSHOT_TABLES:
        # tbl is from the whitelist above, safe to interpolate.
        await db.execute(text(
            f"CREATE TABLE IF NOT EXISTS {tbl}_backup_{suffix} "
            f"AS SELECT * FROM {tbl}"
        ))
    logger.info("snapshot_complete", suffix=suffix, tables=SNAPSHOT_TABLES)
    return suffix


# ═══════════════════════════════════════════════════════
# Advisory lock
# ═══════════════════════════════════════════════════════
async def try_acquire_advisory_lock(db: AsyncSession) -> bool:
    got = (await db.execute(
        text("SELECT pg_try_advisory_lock(:k)"),
        {"k": ADVISORY_LOCK_KEY},
    )).scalar_one()
    return bool(got)


async def release_advisory_lock(db: AsyncSession) -> None:
    try:
        await db.execute(
            text("SELECT pg_advisory_unlock(:k)"),
            {"k": ADVISORY_LOCK_KEY},
        )
        await db.commit()
    except Exception:
        logger.exception("advisory_unlock_failed")


# ═══════════════════════════════════════════════════════
# Constant-time compare
# ═══════════════════════════════════════════════════════
def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())