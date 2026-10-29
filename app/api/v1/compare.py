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
    BaseGroupDetail, GroupDetail,
    GroupHistoryResponse, MatchRunStatusResponse, MatchStatsResponse,
    ProductGroupResponse,
)
from app.services.comparison_service import ComparisonService
from app.services.matching_helpers import constant_time_compare, release_advisory_lock, snapshot_derived_tables, try_acquire_advisory_lock


logger = get_logger(__name__)
router = APIRouter()

_CONFIRM_TOKEN = "YES-DELETE-EVERYTHING"


# ═══════════════════════════════════════════════════════
# READ endpoints
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



# ═══════════════════════════════════════════════════════
# DESTRUCTIVE endpoint
# ═══════════════════════════════════════════════════════
@router.post("/match/run")
async def run_match(
    max_distance: float = Query(0.25, ge=0.0, le=1.0),
    max_candidates: int | None = Query(None, ge=1),
    llm_budget: int | None = Query(None, ge=0),
    clear_existing: bool = Query(
        False,
        description=(
            "DESTRUCTIVE. Truncates canonical_products, base_groups, "
            "product_match_pairs before rebuilding. Requires "
            "confirm=YES-DELETE-EVERYTHING AND the "
            "X-Admin-Destructive-Token header."
        ),
    ),
    confirm: str | None = Query(None),
    sync: bool = Query(False, description="Dev only."),
    x_admin_destructive_token: str | None = Header(
        None, alias="X-Admin-Destructive-Token",
    ),
    user=Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    request_id = getattr(user, "request_id", None)

    # Guard 1: sync forbidden outside dev.
    if sync and settings.environment != "development":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "sync=true is disabled outside development.",
        )

    # Guard 2: destructive confirmation chain.
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
                "DESTRUCTIVE_ADMIN_TOKEN is not configured.",
            )
        if not x_admin_destructive_token:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing X-Admin-Destructive-Token header.",
            )
        if not constant_time_compare(
            x_admin_destructive_token, expected,
        ):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Invalid X-Admin-Destructive-Token.",
            )

    # Guard 3: refuse if products table is suspiciously small.
    if clear_existing:
        n_products = (await db.execute(
            text("SELECT COUNT(*) FROM products")
        )).scalar_one()
        if n_products < 1000:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Refusing destructive run: products has only "
                f"{n_products} rows.",
            )

    # Guard 4: advisory lock.
    if not await try_acquire_advisory_lock(db):
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
        if clear_existing:
            suffix = await snapshot_derived_tables(db)
            await db.commit()
            logger.info("matching_snapshot_done", suffix=suffix)

        # Dev / inline path.
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
                logger.exception(
                    "matching_run_failed", error=str(exc)[:300]
                )
                raise
            return {"status": "completed", "stats": stats.as_dict()}

        # Prod path: enqueue the Celery task.
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
        await release_advisory_lock(db)


@router.get(
    "/match/stats",
    response_model=MatchStatsResponse,
    dependencies=[Depends(require_superadmin)],
)
async def get_match_stats(db: AsyncSession = Depends(get_db)):
    from app.repositories.match_repo import MatchRepo
    stats = await MatchRepo(db).stats()
    # Let the defaults fill in any missing key; do NOT override
    # the real values with hardcoded zeros.
    return MatchStatsResponse(**stats)


@router.get(
    "/match/status",
    response_model=MatchRunStatusResponse,
    dependencies=[Depends(require_superadmin)],
)
async def get_match_status(db: AsyncSession = Depends(get_db)):
    """
    Check whether a matching pass is currently running.

    Reads the advisory lock without acquiring it (pg_locks).
    Returns whether the lock is held and, if so, the PID and the
    number of running matching tasks (best-effort).
    """
    row = (await db.execute(
        text(
            """
            SELECT
                COUNT(*) FILTER (WHERE granted) AS held,
                (ARRAY_AGG(pid)
                    FILTER (WHERE granted))[1]  AS pid
            FROM pg_locks
            WHERE locktype = 'advisory'
              AND objid = :key
            """
        ),
        {"key": 987654321},
    )).first()

    active = bool(row and row.held and int(row.held) > 0)
    return MatchRunStatusResponse(
        active=active,
        progress={"pid": row.pid} if active and row.pid else {},
    )