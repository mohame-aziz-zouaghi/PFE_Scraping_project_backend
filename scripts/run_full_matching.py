"""
Run the full cross-site matching pipeline on the entire catalog.

Usage:
    cd backend
    python -m scripts.run_full_matching
    python -m scripts.run_full_matching --max-candidates 1000
    python -m scripts.run_full_matching --no-clear
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
    from app.db.redis_client import close_redis
    from app.db.session import async_session_maker, dispose_engine
    from app.services.matching_service import MatchingService

    print("=" * 70)
    print("FULL MATCHING RUN")
    print("=" * 70)
    print(f"  Model:              {settings.active_embedding_model}")
    print(f"  LLM enabled:        {settings.matching_llm_enabled}")
    print(f"  Clear existing:     {not args.no_clear}")
    print(f"  Max candidates:     {args.max_candidates or 'unlimited'}")
    print("=" * 70)

    started = time.monotonic()

    try:
        async with async_session_maker() as db:
            svc = MatchingService(db)
            stats = await svc.run_full_match(
                max_candidates=args.max_candidates,
                clear_existing=not args.no_clear,
            )

            elapsed = time.monotonic() - started
            print()
            print("=" * 70)
            print("RESULTS")
            print("=" * 70)
            for k, v in stats.as_dict().items():
                print(f"  {k:.<30} {v}")
            print("=" * 70)
            print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print("=" * 70)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full cross-site matching."
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Limit number of candidate pairs (for testing).",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing groups before running.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())