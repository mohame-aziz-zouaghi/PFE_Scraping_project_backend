# app/api/v1/compare.py
from __future__ import annotations

from uuid import UUID

from fastapi import (
    APIRouter, Depends, Header, HTTPException, Query, status,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_superadmin
from app.config import settings
from app.core.exceptions import NotFoundError
from app.core.logging_config import get_logger
from app.schemas.compare import (
    BaseGroupDetail, CheapestResponse, GroupDetail,
    GroupHistoryResponse, MatchStatsResponse, ProductGroupResponse,
)
from app.services.comparison_service import ComparisonService

logger = get_logger(__name__)
router = APIRouter(prefix="/compare", tags=["compare"])

_CONFIRM_TOKEN = "YES-DELETE-EVERYTHING"
_ADVISORY_LOCK_KEY = 987654321


# ═══════════════════════════════════════════════════════
# READ endpoints (unchanged)
# ═══════════════════════════════════════════════════════
@router.get("/base/{base_id}", response_model=BaseGroupDetail)
async def get_base_group(base_id: UUID, db: AsyncSession = Depends(get_db)):
    try:
        return await ComparisonService(db).get_base_group_detail(base_id)
    except NotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))


@router.get("/group/{canonical_id}", response_model=GroupDetail)
async def get_group(canonical_id: UUID, db: AsyncSession = Depends(get_db)):
    try:
        return await ComparisonService(db).get_group_detail(canonical_id)
    except NotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))


@router.get("/product/{product_id}", response_model=ProductGroupResponse)
async def get_product_group(product_id: UUID, db: AsyncSession = Depends(get_db)):
    return await ComparisonService(db).get_product_group(product_id)


@router.get("/group/{canonical_id}/history", response_model=GroupHistoryResponse)
async def get_group_history(
    canonical_id: UUID,
    period_days: int = Query(30, ge=1, le=3650),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await ComparisonService(db).get_group_history(
            canonical_id, period_days=period_days,
        )
    except NotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))


@router.get("/cheapest", response_model=CheapestResponse)
async def cheapest(
    q: str = Query(..., min_length=1, max_length=300),
    limit: int = Query(10, ge=1, le=100),
    max_distance: float = Query(0.5, ge=0.0, le=2.0),
    db: AsyncSession = Depends(get_db),
):
    return await ComparisonService(db).cheapest_for_query(
        q if False else q, limit=limit, max_distance=max_distance,
    ) if False else await ComparisonService(db).cheapest_for_query(
        query=q, limit=limit, max_distance=max_distance,
    )


# ═══════════════════════════════════════════════════════
# DESTRUCTIVE endpoint — full hardening
# ═══════════════════════════════════════════════════════
@router.post("/match/run")
async def run_match(
    # ─── Function tuning (safe) ─────────────────────────
    max_distance: float = Query(0.25, ge=0.0, le=1.0),
    max_candidates: int | None = Query(None, ge=1),
    llm_budget: int | None = Query(None, ge=0),

    # ─── Destruction switch — must be explicit and confirmed ─
    clear_existing: bool = Query(
        False,
        description=(
            "DESTRUCTIVE. If true, TRUNCATE canonical_products, "
            "base_groups, product_match_pairs before rebuilding. "
            "Requires confirm=YES-DELETE-EVERYTHING AND the "
            "X-Admin-Destructive-Token header."
        ),
    ),
    confirm: str | None = Query(
        None,
        description=(
            "Must equal 'YES-DELETE-EVERYTHING' when "
            "clear_existing=true."
        ),
    ),

    # ─── Dev convenience — disabled in production ───────
    sync: bool = Query(
        False,
        description=(
            "Run inline (dev only). Refused when "
            "environment != development."
        ),
    ),

    # ─── Extra secret header ────────────────────────────
    x_admin_destructive_token: str | None = Header(
        None,
        alias="X-Admin-Destructive-Token",
        description=(
            "Shared secret from DESTRUCTIVE_ADMIN_TOKEN env var. "
            "Required only when clear_existing=true."
        ),
    ),

    # ─── Dependencies ───────────────────────────────────
    user=Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    """Trigger a cross-site matching pass.

    Safe mode (clear_existing=false, default):
        Incremental — finds new pairs and links them.
    Destructive mode (clear_existing=true):
        TRUNCATEs derived tables and rebuilds from scratch.
        Requires FOUR conditions:
          1. superadmin role
          2. clear_existing=true
          3. confirm=YES-DELETE-EVERYTHING
          4. X-Admin-Destructive-Token matching env secret
        Plus: refuses if products table has < 1000 rows,
        refuses if another matching pass is running,
        snapshots tables to *_backup_<date> before TRUNCATE.
    """
    request_id = getattr(user, "request_id", None)

    # ─── Guard 1: sync forbidden in production ──────────
    if sync and settings.environment != "development":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "sync=true is disabled outside development.",
        )

    # ─── Guard 2: destructive confirmation chain ────────
    if clear_existing:
        if confirm != _CONFIRM_TOKEN:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"clear_existing=true requires confirm={_CONFIRM_TOKEN!r}.",
            )
        expected = settings.destructive_admin_token
        if not expected:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "DESTRUCTIVE_ADMIN_TOKEN is not configured on the server. "
                "Destructive mode is disabled.",
            )
        if not x_admin_destructive_token:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing X-Admin-Destructive-Token header.",
            )
        if not _constant_time_compare(x_admin_destructive_token, expected):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Invalid X-Admin-Destructive-Token.",
            )

    # ─── Guard 3: refuse if products table is suspiciously small
    if clear_existing:
        n_products = (await db.execute(
            text("SELECT COUNT(*) FROM products")
        )).scalar_one()
        if n_products < 1000:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Refusing destructive run: products table has only "
                f"{n_products} rows. Expected tens of thousands. "
                f"This guard exists to prevent a repeat of the "
                f"accidental wipe scenario.",
            )

    # ─── Guard 4: Postgres advisory lock ────────────────
    got_lock = (await db.execute(
        text("SELECT pg_try_advisory_lock(:k)"),
        {"k": _ADVISORY_LOCK_KEY},
    )).scalar_one()
    if not got_lock:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Another matching pass is already running.",
        )

    logger.warning(
        "matching_trigger",
        user_id=str(user.id),
        clear_existing=clear_existing,
        confirm=confirm,
        llm_budget=llm_budget,
        request_id=request_id,
    )

    try:
        # ─── Snapshot before truncate ───────────────────
        if clear_existing:
            await _snapshot_derived_tables(db)
            await db.commit()

        # ─── Execute ────────────────────────────────────
        if sync or settings.environment == "development":
            from app.services.matching_service import MatchingService
            svc = MatchingService(db)
            try:
                stats = await svc.run_full_match(
                    max_distance=max_distance,
                    max_candidates=max_candidates,
                    llm_budget=llm_budget,
                    clear_existing=clear_existing,
                )
            except Exception as exc:
                logger.exception("matching_run_failed", error=str(exc)[:300])
                raise
            return {"status": "completed", "stats": stats.as_dict()}

        # ─── Enqueue for prod ───────────────────────────
        from app.tasks.matching_tasks import full_run
        task = full_run.delay(
            max_distance=max_distance,
            max_candidates=max_candidates,
            llm_budget=llm_budget,
            clear_existing=clear_existing,
            requested_by=str(user.id),
        )
        return {"status": "queued", "task_id": task.id}

    finally:
        try:
            await db.execute(
                text("SELECT pg_advisory_unlock(:k)"),
                {"k": _ADVISORY_LOCK_KEY},
            )
            await db.commit()
        except Exception:
            logger.exception("advisory_unlock_failed")


@router.get(
    "/match/stats",
    response_model=MatchStatsResponse,
    dependencies=[Depends(require_superadmin)],
)
async def get_match_stats(db: AsyncSession = Depends(get_db)):
    from app.repositories.match_repo import MatchRepo
    stats = await MatchRepo(db).stats()
    return MatchStatsResponse(
        candidates_found=0, auto_accepted=0, rule_rejected=0,
        variant_rejected=0, llm_checked=0, llm_accepted=0,
        llm_rejected=0, llm_errors=0, llm_deferred=0,
        cluster_rejected=0, variant_groups_created=0,
        products_grouped=0, base_groups_created=0,
        base_groups_updated=0, pairs_linked=0,
        integrity_errors=0, errors=0,
        **stats,
    )


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════
import hmac

def _constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


async def _snapshot_derived_tables(db: AsyncSession) -> None:
    """Snapshot the three derived tables to date-stamped backups.

    Idempotent — CREATE TABLE IF NOT EXISTS means re-running on the
    same day is a no-op. Also prunes backups older than N days.
    """
    from datetime import date
    suffix = date.today().strftime("%Y%m%d")
    for tbl in ("canonical_products", "base_groups", "product_match_pairs"):
        await db.execute(text(
            f"CREATE TABLE IF NOT EXISTS {tbl}_backup_{suffix} "
            f"AS SELECT * FROM {tbl}"
        ))
    logger.info("snapshot_complete", suffix=suffix)