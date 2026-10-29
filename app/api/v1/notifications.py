"""
Notifications API.

    GET  /notifications            — paginated inbox
    GET  /notifications/unread-count
    POST /notifications/{id}/read  — mark one read
    POST /notifications/mark-all-read
    GET  /notifications/preferences
    POST /notifications/preferences — upsert a preference row
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text

from app.api.deps import get_current_user
from app.db import acquire_db
from app.models.user import User

router = APIRouter()


# ── Schemas ─────────────────────────────────────────────
class NotificationOut(BaseModel):
    # ORM mode so Pydantic accepts datetimes / Decimals / etc.
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    title: str
    body: Optional[str] = None
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    product_url: Optional[str] = None
    old_price: Optional[float] = None
    new_price: Optional[float] = None
    discount_pct: Optional[float] = None
    is_read: bool
    created_at: datetime          # ← was `str`; FastAPI serialises to ISO-8601


class PreferenceIn(BaseModel):
    scope: str
    target_product_id: Optional[str] = None
    target_alert_type: Optional[str] = None
    enabled: bool


# ── Endpoints ───────────────────────────────────────────
@router.get("", response_model=list[NotificationOut])
async def list_notifications(
    limit: int = Query(30, ge=1, le=100),
    before_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
):
    async with acquire_db(readonly=True) as db:
        params: dict = {"uid": str(current_user.id), "lim": limit}
        where_before = ""
        if before_id:
            where_before = "AND id < :before"
            params["before"] = before_id
        rows = (await db.execute(
            text(f"""
                SELECT id, kind, title, body,
                       product_id::text AS product_id,
                       product_name, product_url,
                       old_price, new_price, discount_pct,
                       is_read, created_at
                FROM notifications
                WHERE user_id = :uid
                {where_before}
                ORDER BY created_at DESC
                LIMIT :lim
            """),
            params,
        )).all()

    return [NotificationOut(**dict(r._mapping)) for r in rows]


@router.get("/unread-count")
async def unread_count(current_user: User = Depends(get_current_user)):
    async with acquire_db(readonly=True) as db:
        n = (await db.execute(
            text(
                "SELECT COUNT(*) FROM notifications "
                "WHERE user_id = :uid AND is_read = FALSE"
            ),
            {"uid": str(current_user.id)},
        )).scalar_one()
    return {"unread": int(n)}


@router.post("/{notification_id}/read")
async def mark_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
):
    async with acquire_db() as db:
        await db.execute(
            text(
                "UPDATE notifications "
                "SET is_read = TRUE, read_at = now() "
                "WHERE id = :nid AND user_id = :uid"
            ),
            {"nid": notification_id, "uid": str(current_user.id)},
        )
    return {"ok": True}


@router.post("/mark-all-read")
async def mark_all_read(current_user: User = Depends(get_current_user)):
    async with acquire_db() as db:
        r = await db.execute(
            text(
                "UPDATE notifications "
                "SET is_read = TRUE, read_at = now() "
                "WHERE user_id = :uid AND is_read = FALSE"
            ),
            {"uid": str(current_user.id)},
        )
    return {"marked": r.rowcount or 0}


@router.get("/preferences")
async def get_preferences(current_user: User = Depends(get_current_user)):
    async with acquire_db(readonly=True) as db:
        rows = (await db.execute(
            text("""
                SELECT scope, target_product_id::text AS target_product_id,
                       target_alert_type, enabled
                FROM notification_preferences
                WHERE user_id = :uid
            """),
            {"uid": str(current_user.id)},
        )).all()
    return [dict(r._mapping) for r in rows]


@router.post("/preferences")
async def upsert_preference(
    body: PreferenceIn,
    current_user: User = Depends(get_current_user),
):
    if body.scope not in ("global", "product", "type"):
        raise HTTPException(400, "invalid scope")
    if body.scope == "product" and not body.target_product_id:
        raise HTTPException(400, "target_product_id required")
    if body.scope == "type" and not body.target_alert_type:
        raise HTTPException(400, "target_alert_type required")

    async with acquire_db() as db:
        await db.execute(
            text("""
                INSERT INTO notification_preferences
                    (user_id, scope, target_product_id,
                     target_alert_type, enabled, updated_at)
                VALUES
                    (:uid, :scope, :pid, :atype, :en, now())
                ON CONFLICT (user_id, scope, target_product_id, target_alert_type)
                DO UPDATE SET enabled = EXCLUDED.enabled, updated_at = now()
            """),
            {
                "uid":   str(current_user.id),
                "scope": body.scope,
                "pid":   body.target_product_id,
                "atype": body.target_alert_type,
                "en":    body.enabled,
            },
        )
    return {"ok": True}