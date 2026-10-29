# app/tasks/notification_tasks.py
from __future__ import annotations

from typing import Optional

from celery import shared_task
from sqlalchemy import text

from app.core.logging_config import get_logger
from app.db.session import AsyncSessionLocal
from app.services.email_service import EmailService
from app.tasks.async_runner import run_async

logger = get_logger(__name__)

EMAIL_SEVERITY_FILTER = {"high", "medium"}


@shared_task(
    name="notifications.send_user_alert_immediate",
    bind=True, max_retries=3, default_retry_delay=30, acks_late=True,
)
def send_user_alert_immediate(self, *, user_id: str, alert_id: int) -> dict:
    try:
        return run_async(
            _send_immediate_async(user_id=user_id, alert_id=alert_id)
        )
    except Exception as exc:
        logger.exception(
            "send_user_alert_immediate_failed",
            user_id=user_id, alert_id=alert_id, error=str(exc)[:200],
        )
        return {"status": "error", "error": str(exc)[:200]}


async def _send_immediate_async(*, user_id: str, alert_id: int) -> dict:
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("""
                SELECT a.id, a.product_id, a.site, a.alert_type,
                       a.old_value, a.new_value, a.change_percent,
                       a.severity, a.notification_sent,
                       p.name AS product_name, p.url AS product_url
                FROM monitoring_alerts a
                LEFT JOIN products p ON p.id = a.product_id
                WHERE a.id = :aid
            """),
            {"aid": alert_id},
        )).first()

        if row is None:
            return {"status": "skipped", "reason": "alert not found"}

        user = (await db.execute(
            text("""
                SELECT u.id, u.email, u.username, u.is_active,
                       COALESCE(
                           (SELECT np.enabled FROM notification_preferences np
                            WHERE np.user_id = u.id AND np.scope = 'global'
                            LIMIT 1),
                           TRUE
                       ) AS global_enabled,
                       COALESCE(
                           (SELECT np.enabled FROM notification_preferences np
                            WHERE np.user_id = u.id
                              AND np.scope = 'type'
                              AND np.target_alert_type = :atype
                            LIMIT 1),
                           TRUE
                       ) AS type_enabled,
                       COALESCE(
                           (SELECT np.enabled FROM notification_preferences np
                            WHERE np.user_id = u.id
                              AND np.scope = 'product'
                              AND np.target_product_id = :pid
                            LIMIT 1),
                           TRUE
                       ) AS product_enabled
                FROM users u
                WHERE u.id = :uid
            """),
            {
                "uid": user_id,
                "atype": row.alert_type,
                "pid": str(row.product_id),
            },
        )).first()

        if user is None or not user.is_active:
            await _mark_notified(db, [alert_id])
            await db.commit()
            return {"status": "skipped", "reason": "user inactive"}

        opted_in = (
            user.global_enabled
            and user.type_enabled
            and user.product_enabled
        )

        await _write_inbox(db, user_id, row, channel="inapp")

        emailed = False
        if opted_in and row.severity in EMAIL_SEVERITY_FILTER:
            try:
                title, body = _format_alert(row)
                html = _single_email_html(user, title, body, row)
                text_ = _single_email_text(user, title, body, row)
                emailed = await EmailService.send(
                    to_email=user.email,
                    subject=title,
                    html=html,
                    text=text_,
                )
            except Exception as exc:
                logger.warning(
                    "immediate_email_failed",
                    user_id=user_id, alert_id=alert_id,
                    error=str(exc)[:200],
                )

        await _mark_notified(db, [alert_id])
        await db.commit()

        logger.info(
            "alert_immediate_sent",
            user_id=user_id, alert_id=alert_id,
            emailed=emailed, opted_in=opted_in,
        )
        return {
            "status": "ok",
            "emailed": emailed,
            "opted_in": opted_in,
        }


@shared_task(
    name="notifications.send_user_alert_digest",
    bind=True, max_retries=3, default_retry_delay=60, acks_late=True,
)
def send_user_alert_digest(
    self, *, user_id: str, alert_ids: list[int], window_seconds: int = 60,
) -> dict:
    try:
        return run_async(_send_digest_async(
            user_id=user_id,
            alert_ids=alert_ids,
            window_seconds=window_seconds,
        ))
    except Exception as exc:
        logger.exception(
            "send_user_alert_digest_failed",
            user_id=user_id, error=str(exc)[:200],
        )
        return {"status": "error", "error": str(exc)[:200]}


async def _send_digest_async(
    *, user_id: str, alert_ids: list[int], window_seconds: int,
) -> dict:
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            text("""
                SELECT a.id, a.product_id, a.site, a.alert_type,
                       a.old_value, a.new_value, a.change_percent,
                       a.severity, a.notification_sent,
                       p.name AS product_name, p.url AS product_url
                FROM monitoring_alerts a
                LEFT JOIN products p ON p.id = a.product_id
                WHERE a.id = ANY(CAST(:aids AS bigint[]))
                  AND a.notification_sent = FALSE
                ORDER BY a.created_at ASC
            """),
            {"aids": alert_ids},
        )).all()

        if not rows:
            return {"status": "skipped", "reason": "all already notified"}

        user = (await db.execute(
            text("""
                SELECT u.id, u.email, u.username, u.is_active,
                       COALESCE(
                           (SELECT np.enabled FROM notification_preferences np
                            WHERE np.user_id = u.id AND np.scope = 'global'
                            LIMIT 1),
                           TRUE
                       ) AS global_enabled
                FROM users u
                WHERE u.id = :uid
            """),
            {"uid": user_id},
        )).first()

        if user is None or not user.is_active:
            await _mark_notified(db, [r.id for r in rows])
            await db.commit()
            return {"status": "skipped", "reason": "user inactive"}

        for r in rows:
            await _write_inbox(db, user_id, r, channel="email")

        emailed = False
        if user.global_enabled:
            try:
                subject = f"{len(rows)} updates on your wishlist"
                html = _digest_email_html(user, rows)
                text_ = _digest_email_text(user, rows)
                emailed = await EmailService.send(
                    to_email=user.email,
                    subject=subject,
                    html=html,
                    text=text_,
                )
            except Exception as exc:
                logger.warning(
                    "digest_email_failed",
                    user_id=user_id, error=str(exc)[:200],
                )

        await _mark_notified(db, [r.id for r in rows])
        await db.commit()

        logger.info(
            "alert_digest_sent",
            user_id=user_id, alert_count=len(rows), emailed=emailed,
        )
        return {
            "status": "ok",
            "alert_count": len(rows),
            "emailed": emailed,
        }


@shared_task(name="notifications.sweep_unsent_alerts")
def sweep_unsent_alerts(*, max_age_minutes: int = 15) -> dict:
    try:
        return run_async(_sweep_async(max_age_minutes=max_age_minutes))
    except Exception as exc:
        logger.exception(
            "sweep_unsent_alerts_failed", error=str(exc)[:200],
        )
        return {"status": "error", "error": str(exc)[:200]}


async def _sweep_async(*, max_age_minutes: int) -> dict:
    from app.services.alert_fanout import AlertEvent, alert_fanout

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            text("""
                SELECT id, product_id, site, alert_type, severity,
                       old_value, new_value, change_percent
                FROM monitoring_alerts
                WHERE notification_sent = FALSE
                  AND is_resolved = FALSE
                  AND created_at < now() - make_interval(mins => :mins)
                  AND (
                      (alert_type = 'price_drop'
                       AND COALESCE(change_percent, 0) <= -5)
                      OR alert_type = 'back_in_stock'
                  )
                ORDER BY created_at ASC
                LIMIT 500
            """),
            {"mins": max_age_minutes},
        )).all()

    replayed = 0
    for r in rows:
        alert_fanout.enqueue(AlertEvent(
            alert_id=int(r.id),
            product_id=r.product_id,
            site=r.site,
            alert_type=r.alert_type,
            severity=r.severity,
            old_value=r.old_value,
            new_value=r.new_value,
            change_pct=r.change_percent,
        ))
        replayed += 1

    logger.info("sweep_unsent_alerts_replayed", count=replayed)
    return {"status": "ok", "replayed": replayed}


async def _mark_notified(db, alert_ids: list[int]) -> None:
    if not alert_ids:
        return
    await db.execute(
        text("""
            UPDATE monitoring_alerts
            SET notification_sent = TRUE,
                notification_sent_at = now()
            WHERE id = ANY(CAST(:aids AS bigint[]))
        """),
        {"aids": alert_ids},
    )


async def _write_inbox(db, user_id: str, a, *, channel: str) -> int:
    title, body = _format_alert(a)
    try:
        res = await db.execute(
            text("""
                INSERT INTO notifications (
                    user_id, alert_id, kind, title, body,
                    product_id, product_name, product_url,
                    old_price, new_price, discount_pct,
                    channel
                ) VALUES (
                    :uid, :aid, :kind, :title, :body,
                    :pid, :pname, :purl,
                    :oldp, :newp, :disc,
                    :chan
                )
                ON CONFLICT DO NOTHING
            """),
            {
                "uid":  user_id,
                "aid":  int(a.id),
                "kind": (
                    "price_alert" if a.alert_type == "price_drop"
                    else "stock_alert"
                ),
                "title": title,
                "body":  body,
                "pid":  str(a.product_id),
                "pname": (a.product_name or "")[:200],
                "purl": (a.product_url or "")[:500],
                "oldp": _num(a.old_value),
                "newp": _num(a.new_value),
                "disc": a.change_percent,
                "chan": channel,
            },
        )
        return 1 if (res.rowcount or 0) > 0 else 0
    except Exception as exc:
        logger.warning(
            "inbox_write_failed",
            user_id=user_id, alert_id=int(a.id),
            error=str(exc)[:200],
        )
        return 0


def _format_alert(a) -> tuple[str, str]:
    name = (a.product_name or "Product")[:80]
    if a.alert_type == "price_drop":
        title = f"Price drop: {name}"
        try:
            pct = float(a.change_percent or 0.0)
            body = f"{a.old_value} → {a.new_value} ({pct:+.1f}%) on {a.site}"
        except Exception:
            body = f"{a.old_value} → {a.new_value} on {a.site}"
    elif a.alert_type == "back_in_stock":
        title = f"Back in stock: {name}"
        body = f"Available again on {a.site}"
    else:
        title = f"Update: {name}"
        body = f"{a.old_value} → {a.new_value} on {a.site}"
    return title, body


def _num(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except Exception:
        return None


def _single_email_html(user, title, body, a) -> str:
    display = (
        getattr(user, "display_name", None)
        or getattr(user, "username", None)
        or user.email
    )
    product_url = getattr(a, "product_url", None)
    button = (
        f'<p style="margin:24px 0;">'
        f'<a href="{product_url}" '
        f'style="display:inline-block;padding:12px 24px;'
        f'background:#0a66c2;color:#fff;text-decoration:none;'
        f'border-radius:6px;font-weight:600;font-size:14px;">'
        f'View product</a></p>'
        if product_url else ""
    )
    return f"""\
<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#f4f5f7;
font-family:-apple-system,Segoe UI,Roboto,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0">
  <tr><td align="center" style="padding:40px 20px;">
    <table role="presentation" width="560" cellpadding="0" cellspacing="0"
           style="background:#fff;border-radius:8px;
                  box-shadow:0 1px 3px rgba(0,0,0,0.06);overflow:hidden;">
      <tr><td style="padding:32px 40px 8px;">
        <div style="font-size:13px;letter-spacing:.12em;
                    text-transform:uppercase;color:#6b7280;
                    font-weight:600;">Price Monitor</div>
        <h1 style="margin:8px 0 0;font-size:22px;color:#111827;
                   font-weight:700;">{title}</h1>
      </td></tr>
      <tr><td style="padding:16px 40px 32px;color:#374151;
                     font-size:15px;line-height:1.6;">
        <p>Hi {display},</p>
        <p>{body}</p>
        {button}
      </td></tr>
      <tr><td style="padding:16px 40px 32px;border-top:1px solid #f3f4f6;
                     color:#9ca3af;font-size:12px;line-height:1.5;">
        You are receiving this because you asked to be notified about
        products you are interested in. Manage your preferences in
        your account settings.
      </td></tr>
    </table>
  </td></tr>
</table>
</body></html>"""


def _single_email_text(user, title, body, a) -> str:
    display = (
        getattr(user, "display_name", None)
        or getattr(user, "username", None)
        or user.email
    )
    product_url = getattr(a, "product_url", None)
    parts = [f"Hi {display},", "", title, "", body, ""]
    if product_url:
        parts.append(f"View: {product_url}")
        parts.append("")
    parts.append("Manage preferences in your account settings.")
    return "\n".join(parts)


def _digest_email_html(user, alerts) -> str:
    display = (
        getattr(user, "display_name", None)
        or getattr(user, "username", None)
        or user.email
    )
    items = "".join(
        f"<li style='margin-bottom:12px'>"
        f"<strong>{_format_alert(a)[0]}</strong><br>"
        f"<span style='color:#555'>{_format_alert(a)[1]}</span>"
        f"</li>"
        for a in alerts
    )
    return f"""\
<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#f4f5f7;
font-family:-apple-system,Segoe UI,Roboto,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0">
  <tr><td align="center" style="padding:40px 20px;">
    <table role="presentation" width="560" cellpadding="0" cellspacing="0"
           style="background:#fff;border-radius:8px;
                  box-shadow:0 1px 3px rgba(0,0,0,0.06);overflow:hidden;">
      <tr><td style="padding:32px 40px 8px;">
        <div style="font-size:13px;letter-spacing:.12em;
                    text-transform:uppercase;color:#6b7280;
                    font-weight:600;">Price Monitor</div>
        <h1 style="margin:8px 0 0;font-size:22px;color:#111827;
                   font-weight:700;">{len(alerts)} updates on your wishlist</h1>
      </td></tr>
      <tr><td style="padding:16px 40px 32px;color:#374151;
                     font-size:15px;line-height:1.6;">
        <p>Hi {display},</p>
        <ul style="padding-left:20px;margin:16px 0;">{items}</ul>
      </td></tr>
      <tr><td style="padding:16px 40px 32px;border-top:1px solid #f3f4f6;
                     color:#9ca3af;font-size:12px;line-height:1.5;">
        You are receiving this because you asked to be notified about
        products you are interested in. Manage your preferences in
        your account settings.
      </td></tr>
    </table>
  </td></tr>
</table>
</body></html>"""


def _digest_email_text(user, alerts) -> str:
    display = (
        getattr(user, "display_name", None)
        or getattr(user, "username", None)
        or user.email
    )
    lines = [f"Hi {display},", "", f"{len(alerts)} updates on your wishlist:", ""]
    for a in alerts:
        t, b = _format_alert(a)
        lines.append(f"  • {t}")
        lines.append(f"    {b}")
        url = getattr(a, "product_url", None)
        if url:
            lines.append(f"    {url}")
        lines.append("")
    lines.append("Manage preferences in your account settings.")
    return "\n".join(lines)