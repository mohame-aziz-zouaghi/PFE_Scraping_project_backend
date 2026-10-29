"""
Alert rules: interested users + preference filter + online/offline.

One SQL query per batch:

    alert_ids  →  (user_id, alert_id, online: bool)

Includes users who in the last 30 days:
    - wishlisted the product (weight 5 — strongest)
    - clicked the product (weight 2)
    - searched for the product's category or name (weight 5)

Users with an explicit opt-out in notification_preferences are
filtered out at SQL level.

Online = last_seen_at > now() - interval '5 minutes'
Offline = last_seen_at < now() - interval '7 days' OR NULL
In between → treat as online (active session likely)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from sqlalchemy import text

from app.core.logging_config import get_logger

logger = get_logger(__name__)


ONLINE_WINDOW_SECONDS   = 300       # 5 min
OFFLINE_WINDOW_DAYS     = 7
LOOKBACK_DAYS           = 30


@dataclass(slots=True)
class Target:
    user_id:  str
    alert_id: int
    online:   bool


async def resolve_targets(
    db,
    *,
    alert_ids: list[int],
) -> list[Target]:
    """
    Return every (user, alert) pair that should be notified.
    Filters out users with active opt-outs.
    """
    if not alert_ids:
        return []

    rows = (await db.execute(
        text(f"""
            WITH alerts AS (
                SELECT id, product_id
                FROM monitoring_alerts
                WHERE id = ANY(CAST(:aids AS bigint[]))
            ),
            -- Users who wishlisted the product (strongest signal)
            wishlisters AS (
                SELECT DISTINCT wi.user_id, a.id AS alert_id
                FROM wishlist_items wi
                JOIN alerts a ON a.product_id = wi.product_id
            ),
            -- Users who clicked the product in last 30d
            clickers AS (
                SELECT DISTINCT ue.user_id, a.id AS alert_id
                FROM user_events ue
                JOIN alerts a
                    ON (ue.metadata->>'product_id')::uuid = a.product_id
                WHERE ue.event_type = 'click'
                  AND ue.created_at > now() - interval '{LOOKBACK_DAYS} days'
            ),
            -- Users who searched for the product's category/name
            searchers AS (
                SELECT DISTINCT ue.user_id, a.id AS alert_id
                FROM user_events ue
                JOIN alerts a ON true
                JOIN products p ON p.id = a.product_id
                WHERE ue.event_type = 'search'
                  AND ue.created_at > now() - interval '{LOOKBACK_DAYS} days'
                  AND (
                      p.category IS NOT NULL
                      AND (ue.metadata->>'query') ILIKE '%' || p.category || '%'
                  )
            ),
            interested AS (
                SELECT user_id, alert_id FROM wishlisters
                UNION
                SELECT user_id, alert_id FROM clickers
                UNION
                SELECT user_id, alert_id FROM searchers
            )
            SELECT DISTINCT
                i.user_id::text AS user_id,
                i.alert_id,
                CASE
                    WHEN u.last_seen_at IS NULL THEN false
                    WHEN u.last_seen_at > now()
                         - make_interval(secs => {ONLINE_WINDOW_SECONDS})
                        THEN true
                    WHEN u.last_seen_at < now()
                         - make_interval(days => {OFFLINE_WINDOW_DAYS})
                        THEN false
                    ELSE true
                END AS online
            FROM interested i
            JOIN users u ON u.id = i.user_id
            WHERE u.is_active = TRUE
              -- Drop if user opted out of this alert type
              AND NOT EXISTS (
                  SELECT 1 FROM notification_preferences np
                  WHERE np.user_id = i.user_id
                    AND np.enabled = FALSE
                    AND (
                        (np.scope = 'global')
                        OR (np.scope = 'type'
                            AND np.target_alert_type = (
                                SELECT alert_type FROM monitoring_alerts
                                WHERE id = i.alert_id
                            ))
                        OR (np.scope = 'product'
                            AND np.target_product_id = (
                                SELECT product_id FROM monitoring_alerts
                                WHERE id = i.alert_id
                            ))
                    )
              )
        """),
        {"aids": alert_ids},
    )).all()

    targets = [
        Target(user_id=r.user_id, alert_id=int(r.alert_id), online=bool(r.online))
        for r in rows
    ]
    logger.debug(
        "alert_resolve_targets",
        alerts=len(alert_ids),
        targets=len(targets),
        online=sum(1 for t in targets if t.online),
    )
    return targets