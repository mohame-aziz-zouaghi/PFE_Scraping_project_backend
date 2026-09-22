"""
Backfill product embeddings.

Usage:
    cd backend
    python -m scripts.backfill_embeddings --dry-run
    python -m scripts.backfill_embeddings --limit 100
    python -m scripts.backfill_embeddings --site megapc
    python -m scripts.backfill_embeddings                 # full run
    python -m scripts.backfill_embeddings --force         # re-embed everything

Ctrl+C is graceful — stops after the current batch, DB is consistent.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = _THIS_FILE.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


async def _run(args: argparse.Namespace) -> int:
    from app.config import settings
    from app.db.session import async_session_maker, dispose_engine
    from app.db.redis_client import close_redis
    from app.services.embedding_backfill_service import (
        EmbeddingBackfillService,
    )

    model = settings.openai_embedding_model
    print(f"Model:      {model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Site:       {args.site or 'all'}")
    print(f"Force:      {args.force}")
    print(f"Limit:      {args.limit or 'none'}")
    print("─" * 60)

    total_embedded = 0
    total_failed = 0
    offset = 0
    started = time.monotonic()

    try:
        async with async_session_maker() as db:
            svc = EmbeddingBackfillService(db, model=model)

            pending = await svc.count_pending()
            print(f"Pending before run: {pending}")

            if args.dry_run:
                print("Dry run — nothing written.")
                return 0

            if pending == 0 and not args.force:
                print("Nothing to do.")
                return 0

            while True:
                if args.limit and total_embedded >= args.limit:
                    print(f"Reached --limit {args.limit}")
                    break

                stats, offset = await svc.backfill_batch(
                    offset=offset,
                    batch_size=args.batch_size,
                    site=args.site,
                    force=args.force,
                )

                if stats.total_candidates == 0:
                    print("No more candidates.")
                    break

                total_embedded += stats.embedded
                total_failed += stats.failed

                elapsed = time.monotonic() - started
                rate = total_embedded / elapsed if elapsed > 0 else 0
                print(
                    f"  batch done | "
                    f"embedded={total_embedded:>6} "
                    f"failed={total_failed:>4} "
                    f"offset={offset:>6} "
                    f"rate={rate:>5.1f}/s"
                )

                if args.limit and total_embedded >= args.limit:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted. Progress up to the last committed batch "
              "is safe.")
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

    elapsed = time.monotonic() - started
    print("─" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  embedded: {total_embedded}")
    print(f"  failed:   {total_failed}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill embeddings.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
