# app/services/event_service.py
"""Event ingestion with dedup and validation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.models.product import Product
from app.models.user_events import EVENT_WEIGHTS, UserEvent
from app.repositories.event_repo import EventRepo
from app.schemas.recommendation import EventIn, EventsBatchIn

logger = get_logger(__name__)

# Dedup window: skip an event if the same (user, product, type)
# exists within the last N minutes.
_DEDUP_WINDOW_MINUTES = 60

# View and click are deduped. wishlist, add_to_cart and search are
# not — every occurrence is meaningful. dwell is aggregated
# client-side.
_DEDUPED_TYPES = {"view", "click"}


class EventService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = EventRepo(db)

    async def ingest(
        self,
        *,
        user_id: UUID,
        payload: EventsBatchIn,
    ) -> tuple[int, int, list[dict]]:
        """Persist a batch of events. Returns
        (accepted, rejected, errors)."""
        if not payload.events:
            return 0, 0, []

        # ── Filter out events whose product doesn't exist. Search
        #    events are exempt (they have no product FK).
        product_ids = {
            e.product_id for e in payload.events
            if e.product_id is not None
        }
        valid_ids: set[UUID] = set()
        if product_ids:
            result = await self.db.execute(
                select(Product.id).where(Product.id.in_(product_ids))
            )
            valid_ids = {row[0] for row in result.all()}

        accepted_events: list[EventIn] = []
        errors: list[dict] = []
        for idx, e in enumerate(payload.events):
            if e.type == "search":
                accepted_events.append(e)
                continue
            if e.product_id not in valid_ids:
                errors.append({
                    "index": idx,
                    "type": e.type,
                    "product_id": str(e.product_id),
                    "reason": "product_not_found",
                })
                continue
            accepted_events.append(e)

        if not accepted_events:
            return 0, len(errors), errors

        # ── Dedup views and clicks within the window.
        deduped: list[EventIn] = []
        skipped = 0
        for e in accepted_events:
            if e.type in _DEDUPED_TYPES:
                if await self._recently_logged(
                    user_id=user_id,
                    product_id=e.product_id,
                    event_type=e.type,
                ):
                    skipped += 1
                    continue
            deduped.append(e)

        # ── Build rows and insert.
        rows = [
            {
                "event_type": e.type,
                "product_id": e.product_id,
                "metadata": e.metadata,
                "weight": EVENT_WEIGHTS.get(e.type, 1.0),
                "session_id": payload.session_id,
            }
            for e in deduped
        ]
        await self.repo.insert_batch(user_id=user_id, rows=rows)
        await self.db.commit()

        logger.info(
            "events_ingested",
            user_id=str(user_id),
            accepted=len(deduped),
            deduped=skipped,
            invalid=len(errors),
        )
        return len(deduped), skipped + len(errors), errors

    async def _recently_logged(
        self,
        *,
        user_id: UUID,
        product_id: Optional[UUID],
        event_type: str,
    ) -> bool:
        if product_id is None:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=_DEDUP_WINDOW_MINUTES
        )
        stmt = (
            select(func.count())
            .select_from(UserEvent)
            .where(
                UserEvent.user_id == user_id,
                UserEvent.product_id == product_id,
                UserEvent.event_type == event_type,
                UserEvent.created_at >= cutoff,
            )
        )
        count = int((await self.db.execute(stmt)).scalar_one())
        return count > 0