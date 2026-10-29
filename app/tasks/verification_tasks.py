# app/tasks/verification_tasks.py
"""
Celery tasks for the two verification loops.

Both tasks:
  - Respect an LLM daily call budget (RAG.DAILY_CALL_BUDGET).
  - Log every verdict to their respective audit tables.
  - Are idempotent: re-running with the same scope produces the
    same DB state.
  - Isolate failures per-canonical / per-user so a single bad
    record does not abort the batch.

The `_run_with_fresh_engine` wrapper disposes the shared asyncpg
engine inside the same event loop, preventing the recurring
'NoneType object has no attribute send' crash when multiple
Celery tasks run back-to-back.
"""
from __future__ import annotations

import asyncio
from typing import Optional
from uuid import UUID

from celery import shared_task
from sqlalchemy import text

from app.core.logging_config import get_logger
from app.db.session import AsyncSessionLocal

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════
CANONICALS_PER_RUN = 50
RECOMMENDATIONS_PER_RUN = 50
LOW_CONFIDENCE_THRESHOLD = 0.75

# Rematch runs a bulk UPDATE against products. Give it room.
REMATCH_STATEMENT_TIMEOUT_MS = 120_000


# Map the verifier's internal decision names to the DB-allowed
# values in the canonical_verifications CHECK constraint.
_DECISION_MAP = {
    "same": "confirmed",
    "confirmed": "confirmed",
    "split": "split",
    "mismatch": "mismatch",
    "error": "error",
    "budget_exhausted": "budget_exhausted",
}


# ═══════════════════════════════════════════════════════
# Event-loop-safe wrapper
# ═══════════════════════════════════════════════════════
async def _run_with_fresh_engine(coro):
    """
    Wrap an async coroutine so the DB engine is disposed inside
    the SAME event loop.

    Without this, repeated `asyncio.run()` calls in Celery tasks
    hand each other asyncpg connections whose sockets belong to
    a previously closed event loop, producing:
        AttributeError: 'NoneType' object has no attribute 'send'
    """
    from app.db.session import dispose_engine
    try:
        return await coro
    finally:
        try:
            await dispose_engine()
        except Exception as exc:
            logger.warning(
                "engine_dispose_failed", error=str(exc)[:200],
            )


# ═══════════════════════════════════════════════════════
# Task: verify canonicals
# ═══════════════════════════════════════════════════════
@shared_task(name="verification.verify_canonicals")
def verify_canonicals(
    *,
    limit: int = CANONICALS_PER_RUN,
    force_recheck: bool = False,
) -> dict:
    try:
        result = asyncio.run(
            _run_with_fresh_engine(
                _verify_canonicals_async(
                    limit=limit, force_recheck=force_recheck,
                )
            )
        )
        logger.info("verify_canonicals_complete", **result)
        return result
    except Exception as exc:
        logger.exception(
            "verify_canonicals_failed", error=str(exc)[:300],
        )
        return {"status": "error", "error": str(exc)[:200]}


async def _verify_canonicals_async(
    *, limit: int, force_recheck: bool,
) -> dict:
    from app.services.canonical_verifier import CanonicalVerifier

    checked = 0
    confirmed = 0
    mismatches = 0
    splits = 0
    errors = 0
    budget_hit = 0
    rematches_queued = 0
    skipped = 0

    async with AsyncSessionLocal() as db:
        candidates = await _pick_canonical_candidates(
            db, limit=limit, force_recheck=force_recheck,
        )

        for canonical_id, base_group_id, _confidence in candidates:
            # ── Budget check (fail-closed) ─────────────────────
            if await _daily_budget_exceeded(db):
                budget_hit += 1
                await _safe_write_canonical_verdict(
                    db,
                    canonical_id=canonical_id,
                    base_group_id=base_group_id,
                    decision="budget_exhausted",
                    reason="daily LLM budget exhausted",
                )
                await _safe_commit(db)
                continue

            # ── Verify (per-canonical try/except) ──────────────
            try:
                verifier = CanonicalVerifier(db)
                verdict = await verifier.verify(
                    canonical_id=canonical_id, timeout_s=120,
                )
            except Exception as exc:
                errors += 1
                logger.warning(
                    "canonical_verify_row_failed",
                    canonical_id=str(canonical_id),
                    error=str(exc)[:300],
                )
                await _safe_write_canonical_verdict(
                    db,
                    canonical_id=canonical_id,
                    base_group_id=base_group_id,
                    decision="error",
                    reason=f"{type(exc).__name__}: {str(exc)[:180]}",
                    error_detail=str(exc)[:800],
                )
                await _safe_commit(db)
                continue

            checked += 1

            # ── Read fresh counters for the audit row ──────────
            listing_count = 0
            site_count = 0
            try:
                row = (await db.execute(
                    text(
                        "SELECT listing_count, site_count "
                        "FROM canonical_products WHERE id = :cid"
                    ),
                    {"cid": str(canonical_id)},
                )).first()
                if row:
                    listing_count = int(row.listing_count or 0)
                    site_count = int(row.site_count or 0)
            except Exception:
                # Don't fail the row for a counter read
                await _safe_rollback(db)

            rematch_queued = False
            rematch_task_id = None

            if verdict.decision in ("mismatch", "split"):
                if verdict.decision == "split":
                    splits += 1
                else:
                    mismatches += 1

                try:
                    task = rematch_canonical.delay(
                        canonical_id=str(canonical_id),
                        base_group_id=(
                            str(base_group_id)
                            if base_group_id else None
                        ),
                        outlier_listing_ids=[
                            str(x) for x in verdict.outlier_listing_ids
                        ],
                        reason=verdict.reason[:200],
                    )
                    rematch_queued = True
                    rematch_task_id = task.id
                    rematches_queued += 1
                except Exception as exc:
                    logger.warning(
                        "rematch_queue_failed",
                        canonical_id=str(canonical_id),
                        error=str(exc)[:200],
                    )
                    # Reset to pending so next cycle retries
                    try:
                        await db.execute(
                            text(
                                "UPDATE canonical_products "
                                "SET verification_state = 'pending' "
                                "WHERE id = :cid"
                            ),
                            {"cid": str(canonical_id)},
                        )
                        await _safe_commit(db)
                    except Exception:
                        await _safe_rollback(db)
            elif verdict.decision == "same":
                confirmed += 1
            elif verdict.decision == "error":
                errors += 1
            else:
                skipped += 1

            db_decision = _DECISION_MAP.get(
                verdict.decision, "error"
            )

            await _safe_write_canonical_verdict(
                db,
                canonical_id=canonical_id,
                base_group_id=base_group_id,
                decision=db_decision,
                confidence=verdict.confidence,
                reason=verdict.reason,
                error_detail=(
                    verdict.reason
                    if verdict.decision == "error" else None
                ),
                listing_count=listing_count,
                site_count=site_count,
                rematch_queued=rematch_queued,
                rematch_task_id=rematch_task_id,
                llm_model=verdict.llm_model,
                llm_latency_ms=verdict.llm_latency_ms,
                prompt_tokens=verdict.prompt_tokens,
                completion_tokens=verdict.completion_tokens,
            )
            await _safe_commit(db)

        return {
            "status": "ok",
            "checked": checked,
            "confirmed": confirmed,
            "mismatches": mismatches,
            "splits": splits,
            "errors": errors,
            "budget_hit": budget_hit,
            "rematches_queued": rematches_queued,
            "skipped": skipped,
        }


# ═══════════════════════════════════════════════════════
# Task: verify recommendations
# ═══════════════════════════════════════════════════════
@shared_task(name="verification.verify_recommendations")
def verify_recommendations(
    *,
    limit: int = RECOMMENDATIONS_PER_RUN,
    force_recheck: bool = False,
    rebuild_on_mismatch: bool = True,
) -> dict:
    try:
        result = asyncio.run(
            _run_with_fresh_engine(
                _verify_recommendations_async(
                    limit=limit,
                    force_recheck=force_recheck,
                    rebuild_on_mismatch=rebuild_on_mismatch,
                )
            )
        )
        logger.info("verify_recommendations_complete", **result)
        return result
    except Exception as exc:
        logger.exception(
            "verify_recommendations_failed", error=str(exc)[:300],
        )
        return {"status": "error", "error": str(exc)[:200]}


async def _verify_recommendations_async(
    *, limit: int, force_recheck: bool,
    rebuild_on_mismatch: bool,
) -> dict:
    from app.services.recommendation_verifier import (
        RecommendationVerifier,
    )

    checked = 0
    confirmed = 0
    mismatches = 0
    rebuilt = 0
    errors = 0
    budget_hit = 0

    async with AsyncSessionLocal() as db:
        candidates = await _pick_recommendation_candidates(
            db, limit=limit, force_recheck=force_recheck,
        )

        for user_id in candidates:
            if await _daily_budget_exceeded(db):
                budget_hit += 1
                await _safe_write_recommendation_verdict(
                    db,
                    user_id=user_id,
                    decision="budget_exhausted",
                    reason="daily LLM budget exhausted",
                )
                await _safe_commit(db)
                continue

            # Per-user try/except
            try:
                verifier = RecommendationVerifier(db)
                verdict = await verifier.verify(
                    user_id=user_id, timeout_s=120,
                )
            except Exception as exc:
                errors += 1
                logger.warning(
                    "recommendation_verify_row_failed",
                    user_id=str(user_id),
                    error=str(exc)[:300],
                )
                await _safe_write_recommendation_verdict(
                    db,
                    user_id=user_id,
                    decision="error",
                    reason=f"{type(exc).__name__}: {str(exc)[:180]}",
                    error_detail=str(exc)[:800],
                )
                await _safe_commit(db)
                continue

            checked += 1

            did_rebuild = False
            if verdict.decision == "confirmed":
                confirmed += 1
            elif verdict.decision == "mismatch":
                mismatches += 1
                if rebuild_on_mismatch:
                    try:
                        from app.services.recommendation_service import (
                            RecommendationService,
                        )
                        await db.execute(
                            text(
                                "DELETE FROM recommendations "
                                "WHERE user_id = :uid"
                            ),
                            {"uid": str(user_id)},
                        )
                        await _safe_commit(db)

                        svc = RecommendationService(db)
                        await svc.for_user(user_id=user_id, limit=15)
                        did_rebuild = True
                        rebuilt += 1
                    except Exception as exc:
                        logger.warning(
                            "rec_verify_rebuild_failed",
                            user_id=str(user_id),
                            error=str(exc)[:200],
                        )
                        await _safe_rollback(db)
            elif verdict.decision == "error":
                errors += 1

            await _safe_write_recommendation_verdict(
                db,
                user_id=user_id,
                decision=verdict.decision,
                confidence=verdict.confidence,
                reason=verdict.reason,
                error_detail=(
                    verdict.reason
                    if verdict.decision == "error" else None
                ),
                item_count=verdict.item_count,
                mismatch_count=len(verdict.mismatched_product_ids),
                rebuilt=did_rebuild,
                llm_model=verdict.llm_model,
                llm_latency_ms=verdict.llm_latency_ms,
                prompt_tokens=verdict.prompt_tokens,
                completion_tokens=verdict.completion_tokens,
            )
            await _safe_commit(db)

        return {
            "status": "ok",
            "checked": checked,
            "confirmed": confirmed,
            "mismatches": mismatches,
            "rebuilt": rebuilt,
            "errors": errors,
            "budget_hit": budget_hit,
        }


# ═══════════════════════════════════════════════════════
# Task: rematch a specific canonical
# ═══════════════════════════════════════════════════════
@shared_task(name="verification.rematch_canonical")
def rematch_canonical(
    *,
    canonical_id: str,
    base_group_id: Optional[str] = None,
    outlier_listing_ids: Optional[list[str]] = None,
    reason: str = "",
) -> dict:
    """
    Two modes:

      SPLIT   (outlier_listing_ids non-empty)
        Detach only those listings. The canonical keeps the rest
        and is marked 'settled' (already verified correct).

      MISMATCH (outlier_listing_ids empty)
        Detach all listings. Delete the canonical. All products
        re-matched from scratch.
    """
    try:
        result = asyncio.run(
            _run_with_fresh_engine(
                _rematch_canonical_async(
                    canonical_id=UUID(canonical_id),
                    base_group_id=(
                        UUID(base_group_id) if base_group_id else None
                    ),
                    outlier_listing_ids=(
                        [UUID(x) for x in outlier_listing_ids]
                        if outlier_listing_ids else None
                    ),
                    reason=reason,
                )
            )
        )
        logger.info("rematch_canonical_complete", **result)
        return result
    except Exception as exc:
        logger.exception(
            "rematch_canonical_failed",
            canonical_id=canonical_id,
            error=str(exc)[:300],
        )
        return {"status": "error", "error": str(exc)[:200]}


async def _rematch_canonical_async(
    *,
    canonical_id: UUID,
    base_group_id: Optional[UUID],
    outlier_listing_ids: Optional[list[UUID]],
    reason: str,
) -> dict:
    from app.services.matching_helpers import (
        recompute_base_cheapest,
        recompute_canonical_cheapest,
    )
    from app.services.matching_service import MatchingService

    async with AsyncSessionLocal() as db:
        # Raise statement_timeout for this transaction only — the
        # bulk detach + rematch can take minutes on large batches.
        await db.execute(text(
            f"SET LOCAL statement_timeout = {REMATCH_STATEMENT_TIMEOUT_MS}"
        ))

        # ── 1. Determine which products to detach
        if outlier_listing_ids:
            to_detach = [str(x) for x in outlier_listing_ids]
            keep_in_canonical = True
        else:
            rows = (await db.execute(
                text(
                    "SELECT id FROM products WHERE canonical_id = :cid"
                ),
                {"cid": str(canonical_id)},
            )).all()
            to_detach = [str(r.id) for r in rows]
            keep_in_canonical = False

        if not to_detach:
            return {"status": "skipped", "reason": "no products to detach"}

        # ── 2. Detach products
        await db.execute(
            text(
                "UPDATE products SET canonical_id = NULL "
                "WHERE id = ANY(CAST(:ids AS uuid[]))"
            ),
            {"ids": to_detach},
        )

        # ── 3. Update canonical/base state
        if not keep_in_canonical:
            await db.execute(
                text("DELETE FROM canonical_products WHERE id = :cid"),
                {"cid": str(canonical_id)},
            )
            if base_group_id:
                await db.execute(
                    text(
                        "DELETE FROM base_groups WHERE id = :bid "
                        "AND NOT EXISTS ("
                        "  SELECT 1 FROM canonical_products "
                        "  WHERE base_group_id = :bid"
                        ")"
                    ),
                    {"bid": str(base_group_id)},
                )
        else:
            await db.execute(
                text("""
                    UPDATE canonical_products cp
                    SET listing_count = sub.listing_count,
                        site_count = sub.site_count,
                        verification_state = 'pending',
                        verified_at = now()
                    FROM (
                        SELECT
                            COUNT(*) AS listing_count,
                            COUNT(DISTINCT site) AS site_count
                        FROM products
                        WHERE canonical_id = :cid
                    ) sub
                    WHERE cp.id = :cid
                """),
                {"cid": str(canonical_id)},
            )
            if base_group_id:
                await db.execute(
                    text("""
                        UPDATE base_groups bg
                        SET variant_count = (
                            SELECT COUNT(*) FROM canonical_products cp
                            WHERE cp.base_group_id = bg.id
                        )
                        WHERE bg.id = :bid
                    """),
                    {"bid": str(base_group_id)},
                )

        await db.commit()

        # ── 4. Delete canonical if split left it empty
        if keep_in_canonical:
            remaining = (await db.execute(
                text(
                    "SELECT COUNT(*) FROM products "
                    "WHERE canonical_id = :cid"
                ),
                {"cid": str(canonical_id)},
            )).scalar_one()
            if int(remaining) == 0:
                logger.info(
                    "rematch_canonical_emptied",
                    canonical_id=str(canonical_id),
                )
                await db.execute(
                    text(
                        "DELETE FROM canonical_products "
                        "WHERE id = :cid"
                    ),
                    {"cid": str(canonical_id)},
                )
                if base_group_id:
                    await db.execute(
                        text(
                            "DELETE FROM base_groups WHERE id = :bid "
                            "AND NOT EXISTS ("
                            "  SELECT 1 FROM canonical_products "
                            "  WHERE base_group_id = :bid"
                            ")"
                        ),
                        {"bid": str(base_group_id)},
                    )
                await db.commit()

        # ── 5. Re-match the detached products
        product_ids = [UUID(x) for x in to_detach]
        svc = MatchingService(db)
        stats = await svc.run_targeted_match(product_ids=product_ids)

        # ── 6. Recompute prices for affected canonicals
        await recompute_canonical_cheapest(db)
        await recompute_base_cheapest(db)
        await db.commit()

        # ── 7. Post-check integrity (informational)
        try:
            integrity = (await db.execute(text(
                "SELECT "
                "  (SELECT COUNT(*) FROM canonical_products "
                "   WHERE base_group_id IS NULL AND listing_count > 0) "
                "+ (SELECT COUNT(*) FROM canonical_products cp "
                "   WHERE cp.listing_count <> ("
                "     SELECT COUNT(*) FROM products p "
                "     WHERE p.canonical_id = cp.id)) "
                "+ (SELECT COUNT(*) FROM base_groups bg "
                "   WHERE bg.variant_count <> ("
                "     SELECT COUNT(*) FROM canonical_products cp "
                "     WHERE cp.base_group_id = bg.id))"
            ))).scalar_one()
            if integrity > 0:
                logger.warning(
                    "rematch_integrity_drift",
                    canonical_id=str(canonical_id),
                    integrity_errors=int(integrity),
                )
        except Exception:
            pass

        logger.info(
            "rematch_canonical_done",
            original_canonical_id=str(canonical_id),
            detached_count=len(product_ids),
            split_mode=keep_in_canonical,
            reason=reason[:200],
            stats=stats.as_dict(),
        )
        return {
            "status": "ok",
            "detached_count": len(product_ids),
            "split_mode": keep_in_canonical,
            "stats": stats.as_dict(),
        }


# ═══════════════════════════════════════════════════════
# Candidate selection
# ═══════════════════════════════════════════════════════
async def _pick_canonical_candidates(
    db, *, limit: int, force_recheck: bool,
) -> list[tuple[UUID, Optional[UUID], float]]:
    state_filter = (
        ""
        if force_recheck
        else "AND cp.verification_state = 'pending'"
    )

    sql = text(f"""
        SELECT
            cp.id AS canonical_id,
            cp.base_group_id,
            COALESCE(cp.confidence, 0.5) AS confidence
        FROM canonical_products cp
        WHERE cp.listing_count >= 2
          AND (cp.confidence IS NULL OR cp.confidence < :threshold)
          {state_filter}
        ORDER BY
            CASE WHEN cp.confidence IS NULL THEN 0 ELSE 1 END ASC,
            COALESCE(cp.confidence, 0) ASC,
            cp.updated_at DESC
        LIMIT :lim
    """)

    rows = (await db.execute(sql, {
        "threshold": LOW_CONFIDENCE_THRESHOLD,
        "lim": limit,
    })).all()

    return [
        (r.canonical_id, r.base_group_id, float(r.confidence))
        for r in rows
    ]


async def _pick_recommendation_candidates(
    db, *, limit: int, force_recheck: bool,
) -> list[UUID]:
    since_filter = ""
    if not force_recheck:
        since_filter = (
            "AND (v.last_verified_at IS NULL "
            "   OR v.last_verified_at < now() - interval '7 days')"
        )

    sql = text(f"""
        SELECT r.user_id
        FROM recommendations r
        INNER JOIN user_profiles up ON up.user_id = r.user_id
        LEFT JOIN (
            SELECT user_id, MAX(created_at) AS last_verified_at
            FROM recommendation_verifications
            WHERE decision IN ('confirmed', 'mismatch', 'rebuild')
            GROUP BY user_id
        ) v ON v.user_id = r.user_id
        WHERE up.is_cold_start = FALSE
          {since_filter}
        GROUP BY r.user_id
        HAVING COUNT(r.id) >= 5
        ORDER BY MAX(r.generated_at) DESC
        LIMIT :lim
    """)

    rows = (await db.execute(sql, {"lim": limit})).all()
    return [r.user_id for r in rows]


# ═══════════════════════════════════════════════════════
# Safe commit/rollback wrappers
# ═══════════════════════════════════════════════════════
async def _safe_commit(db) -> None:
    try:
        await db.commit()
    except Exception as exc:
        logger.warning("safe_commit_failed", error=str(exc)[:200])
        await _safe_rollback(db)


async def _safe_rollback(db) -> None:
    try:
        await db.rollback()
    except Exception as exc:
        logger.warning("safe_rollback_failed", error=str(exc)[:200])


async def _safe_write_canonical_verdict(db, **kwargs) -> None:
    try:
        await _write_canonical_verdict(db, **kwargs)
    except Exception as exc:
        logger.warning(
            "canonical_verdict_safe_write_failed",
            error=str(exc)[:200],
        )
        await _safe_rollback(db)


async def _safe_write_recommendation_verdict(db, **kwargs) -> None:
    try:
        await _write_recommendation_verdict(db, **kwargs)
    except Exception as exc:
        logger.warning(
            "recommendation_verdict_safe_write_failed",
            error=str(exc)[:200],
        )
        await _safe_rollback(db)


# ═══════════════════════════════════════════════════════
# Audit writes
# ═══════════════════════════════════════════════════════
async def _write_canonical_verdict(
    db,
    *,
    canonical_id: UUID,
    base_group_id: Optional[UUID],
    decision: str,
    confidence: float = 0.0,
    reason: str = "",
    error_detail: Optional[str] = None,
    listing_count: int = 0,
    site_count: int = 0,
    rematch_queued: bool = False,
    rematch_task_id: Optional[str] = None,
    llm_model: str = "",
    llm_latency_ms: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    await db.execute(
        text(
            "INSERT INTO canonical_verifications ("
            "  canonical_id, base_group_id, decision,"
            "  confidence, reason, error_detail,"
            "  listing_count, site_count,"
            "  rematch_queued, rematch_task_id,"
            "  llm_model, llm_latency_ms,"
            "  prompt_tokens, completion_tokens"
            ") VALUES ("
            "  :cid, :bid, :dec,"
            "  :conf, :reason, :err,"
            "  :lc, :sc,"
            "  :rq, :rt,"
            "  :model, :lat,"
            "  :pt, :ct"
            ")"
        ),
        {
            "cid": str(canonical_id),
            "bid": str(base_group_id) if base_group_id else None,
            "dec": decision,
            "conf": confidence,
            "reason": (reason or "")[:400],
            "err": (error_detail or "")[:800] if error_detail else None,
            "lc": listing_count,
            "sc": site_count,
            "rq": rematch_queued,
            "rt": rematch_task_id,
            "model": llm_model or "",
            "lat": llm_latency_ms or 0,
            "pt": prompt_tokens or 0,
            "ct": completion_tokens or 0,
        },
    )


async def _write_recommendation_verdict(
    db,
    *,
    user_id: UUID,
    decision: str,
    confidence: float = 0.0,
    reason: str = "",
    error_detail: Optional[str] = None,
    item_count: int = 0,
    mismatch_count: int = 0,
    rebuilt: bool = False,
    llm_model: str = "",
    llm_latency_ms: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    await db.execute(
        text(
            "INSERT INTO recommendation_verifications ("
            "  user_id, decision, confidence, reason,"
            "  error_detail,"
            "  item_count, mismatch_count, rebuilt,"
            "  llm_model, llm_latency_ms,"
            "  prompt_tokens, completion_tokens"
            ") VALUES ("
            "  :uid, :dec, :conf, :reason,"
            "  :err,"
            "  :ic, :mc, :rb,"
            "  :model, :lat,"
            "  :pt, :ct"
            ")"
        ),
        {
            "uid": str(user_id),
            "dec": decision,
            "conf": confidence,
            "reason": (reason or "")[:400],
            "err": (error_detail or "")[:800] if error_detail else None,
            "ic": item_count or 0,
            "mc": mismatch_count or 0,
            "rb": rebuilt,
            "model": llm_model or "",
            "lat": llm_latency_ms or 0,
            "pt": prompt_tokens or 0,
            "ct": completion_tokens or 0,
        },
    )


# ═══════════════════════════════════════════════════════
# Budget check — fail-closed
# ═══════════════════════════════════════════════════════
async def _daily_budget_exceeded(db) -> bool:
    """
    Return True if we've hit the daily LLM call budget.

    Fails CLOSED: if the check itself errors (llm_usage table
    missing, connection dropped), we treat it as "budget hit"
    so we don't silently incur unbounded cost.
    """
    try:
        from app.core.rag_config import RAG
        row = (await db.execute(text(
            "SELECT COUNT(*) FROM llm_usage "
            "WHERE endpoint LIKE 'verify_%' "
            "  AND created_at >= date_trunc('day', now())"
        ))).scalar_one()
        return int(row) >= RAG.DAILY_CALL_BUDGET
    except Exception as exc:
        logger.warning(
            "budget_check_failed_failing_closed",
            error=str(exc)[:200],
        )
        # If the check is unavailable, don't spend LLM money
        # blind. Rollback the broken transaction so the caller
        # can continue with an audit-only verdict.
        await _safe_rollback(db)
        return True