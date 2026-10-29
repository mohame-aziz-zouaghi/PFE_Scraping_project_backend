"""
Rebuild all canonicals and base groups from scratch using the
current identity_signature_hash logic.

Usage:
    cd backend
    python -m scripts.rebuild_all_canonicals
    python -m scripts.rebuild_all_canonicals --batch-size 500
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
BACKEND_ROOT = _THIS.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


async def _run(args) -> int:
    from sqlalchemy import select
    from app.db.session import (
        AsyncSessionLocal,
        dispose_engine,
    )
    from app.db.redis_client import close_redis
    from app.models.product import Product
    from app.services.canonical_rebuilder import CanonicalRebuilder

    started = time.monotonic()
    print("=" * 60)
    print("REBUILD ALL CANONICALS + BASES")
    print("=" * 60)
    print(f"Batch size: {args.batch_size}")
    print()

    try:
        async with AsyncSessionLocal() as db:
            # Load all enabled products
            rows = (await db.execute(
                select(Product.id).where(
                    Product.is_enabled.is_(True)
                )
            )).scalars().all()
            total = len(rows)
            print(f"Total products: {total}")

            # Process in batches
            rebuilt = 0
            canonicals_created = 0
            bases_created = 0

            for start in range(0, total, args.batch_size):
                chunk = rows[start:start + args.batch_size]
                rebuilder = CanonicalRebuilder(db)
                stats = await rebuilder.rebuild_for_products(chunk)
                rebuilt += stats["products"]
                canonicals_created += stats["canonicals_created"]
                bases_created += stats["bases_created"]
                print(
                    f"  {rebuilt}/{total} products "
                    f"| canonicals: {canonicals_created} "
                    f"| bases: {bases_created}"
                )

            elapsed = time.monotonic() - started
            print()
            print(f"Rebuild complete in {elapsed:.1f}s")
            print(f"  Products:    {rebuilt}")
            print(f"  Canonicals:  {canonicals_created}")
            print(f"  Bases:       {bases_created}")

            # Verification
            from sqlalchemy import text
            print()
            print("VERIFICATION")
            print("-" * 60)

            for label, q in (
                ("Base groups",
                 "SELECT COUNT(*) FROM base_groups"),
                ("Canonicals",
                 "SELECT COUNT(*) FROM canonical_products"),
                ("Products with canonical",
                 "SELECT COUNT(*) FROM products "
                 "WHERE is_enabled = TRUE AND canonical_id IS NOT NULL"),
                ("Products without canonical",
                 "SELECT COUNT(*) FROM products "
                 "WHERE is_enabled = TRUE AND canonical_id IS NULL"),
                ("Canonicals with >1 site",
                 "SELECT COUNT(*) FROM canonical_products "
                 "WHERE site_count > 1"),
                ("Canonicals with 1 site",
                 "SELECT COUNT(*) FROM canonical_products "
                 "WHERE site_count = 1"),
            ):
                v = await db.execute(text(q))
                print(f"  {label:30} {v.scalar_one()}")

    except Exception as exc:
        print(f"Failed: {exc}")
        return 1
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())