"""Analytics business logic."""
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product import (
    MonitoringAlert,
    MonitoringBatch,
    PriceHistory,
    Product,
)
from app.schemas.analytics import (
    AlertSummary,
    AvailabilityTimelinePoint,
    AvailabilityTimelineResponse,
    PriceDrop,
    PriceDropsResponse,
    PriceTrendPoint,
    PriceTrendResponse,
    SiteOverview,
    SitesOverviewResponse,
    VolatileProduct,
    VolatileProductsResponse,
)


class AnalyticsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── Price trends ───────────────────────────────────
    async def get_price_trends(
        self,
        site: Optional[str],
        product_id: Optional[str],
        period_days: int,
    ) -> PriceTrendResponse:
        since = datetime.now(timezone.utc) - timedelta(days=period_days)

        stmt = (
            select(
                func.date_trunc("day", PriceHistory.recorded_at).label("day"),
                func.avg(PriceHistory.price).label("avg_price"),
                func.min(PriceHistory.price).label("min_price"),
                func.max(PriceHistory.price).label("max_price"),
                func.count().label("count"),
            )
            .join(Product, Product.id == PriceHistory.product_id)
            .where(PriceHistory.recorded_at >= since)
            .group_by("day")
            .order_by("day")
        )
        if site:
            stmt = stmt.where(Product.site == site.lower())
        if product_id:
            stmt = stmt.where(Product.id == product_id)

        rows = (await self.db.execute(stmt)).all()

        points = [
            PriceTrendPoint(
                date=r.day.date() if hasattr(r.day, "date") else r.day,
                avg_price=float(r.avg_price or 0),
                min_price=float(r.min_price or 0),
                max_price=float(r.max_price or 0),
                count=int(r.count or 0),
            )
            for r in rows
        ]

        return PriceTrendResponse(
            product_id=product_id,
            site=site,
            period_days=period_days,
            points=points,
        )

    # ─── Volatile products ──────────────────────────────
    async def get_volatile_products(
        self,
        site: Optional[str],
        period_days: int,
        limit: int,
    ) -> VolatileProductsResponse:
        since = datetime.now(timezone.utc) - timedelta(days=period_days)

        stmt = (
            select(
                Product.id,
                Product.site,
                Product.name,
                Product.url,
                Product.price,
                Product.price_volatility_score,
                Product.price_change_count,
                Product.last_price_change,
                func.count(PriceHistory.id).label("changes_in_period"),
            )
            .join(PriceHistory, PriceHistory.product_id == Product.id)
            .where(PriceHistory.recorded_at >= since)
            .where(Product.is_enabled == True)
            .group_by(
                Product.id,
                Product.site,
                Product.name,
                Product.url,
                Product.price,
                Product.price_volatility_score,
                Product.price_change_count,
                Product.last_price_change,
            )
            .order_by(desc("changes_in_period"), desc(Product.price_volatility_score))
            .limit(limit)
        )
        if site:
            stmt = stmt.where(Product.site == site.lower())

        rows = (await self.db.execute(stmt)).all()

        items = [
            VolatileProduct(
                product_id=r.id,
                site=r.site,
                name=r.name,
                url=r.url,
                price=float(r.price or 0),
                price_volatility_score=int(r.price_volatility_score or 0),
                price_change_count=int(r.price_change_count or 0),
                last_price_change=r.last_price_change,
            )
            for r in rows
        ]
        return VolatileProductsResponse(
            period_days=period_days, items=items, total=len(items)
        )

    # ─── Biggest price drops ────────────────────────────
    async def get_price_drops(
        self,
        site: Optional[str],
        period_days: int,
        min_change_percent: float,
        limit: int,
    ) -> PriceDropsResponse:
        since = datetime.now(timezone.utc) - timedelta(days=period_days)

        stmt = (
            select(MonitoringAlert, Product)
            .join(Product, Product.id == MonitoringAlert.product_id)
            .where(MonitoringAlert.alert_type == "price_drop")
            .where(MonitoringAlert.created_at >= since)
            .where(MonitoringAlert.change_percent <= -abs(min_change_percent))
            .order_by(MonitoringAlert.change_percent.asc())
            .limit(limit)
        )
        if site:
            stmt = stmt.where(MonitoringAlert.site == site.lower())

        rows = (await self.db.execute(stmt)).all()

        items = []
        for alert, product in rows:
            try:
                old_price = float(alert.old_value or 0)
                new_price = float(alert.new_value or 0)
            except (TypeError, ValueError):
                old_price, new_price = 0.0, 0.0

            items.append(
                PriceDrop(
                    product_id=product.id,
                    site=product.site,
                    name=product.name,
                    url=product.url,
                    old_price=old_price,
                    new_price=new_price,
                    change_percent=float(alert.change_percent or 0),
                    detected_at=alert.created_at,
                )
            )
        return PriceDropsResponse(
            period_days=period_days, items=items, total=len(items)
        )

    # ─── Alerts summary (single query, aggregate in Python) ───
    async def get_alert_summary(
        self, site: Optional[str], period_days: int
    ) -> AlertSummary:
        since = datetime.now(timezone.utc) - timedelta(days=period_days)

        conditions = [MonitoringAlert.created_at >= since]
        if site:
            conditions.append(MonitoringAlert.site == site.lower())

        rows = (await self.db.execute(
            select(
                MonitoringAlert.site,
                MonitoringAlert.alert_type,
                MonitoringAlert.severity,
                MonitoringAlert.is_resolved,
                func.count().label("cnt"),
            )
            .where(and_(*conditions))
            .group_by(
                MonitoringAlert.site,
                MonitoringAlert.alert_type,
                MonitoringAlert.severity,
                MonitoringAlert.is_resolved,
            )
        )).all()

        total = 0
        unresolved = 0
        resolved = 0
        by_site: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}

        for row in rows:
            c = int(row.cnt or 0)
            total += c
            if row.is_resolved:
                resolved += c
            else:
                unresolved += c

            s = row.site or "unknown"
            t = row.alert_type or "unknown"
            sev = row.severity or "unknown"
            by_site[s] = by_site.get(s, 0) + c
            by_type[t] = by_type.get(t, 0) + c
            by_severity[sev] = by_severity.get(sev, 0) + c

        return AlertSummary(
            total=total,
            unresolved=unresolved,
            resolved=resolved,
            by_site=by_site,
            by_type=by_type,
            by_severity=by_severity,
        )

    # ─── Availability timeline ──────────────────────────
    async def get_availability_timeline(
        self, site: Optional[str], period_days: int
    ) -> AvailabilityTimelineResponse:
        since = datetime.now(timezone.utc) - timedelta(days=period_days)

        stmt = (
            select(
                func.date_trunc("day", PriceHistory.recorded_at).label("day"),
                PriceHistory.availability,
                func.count().label("count"),
            )
            .join(Product, Product.id == PriceHistory.product_id)
            .where(PriceHistory.recorded_at >= since)
            .group_by("day", PriceHistory.availability)
            .order_by("day")
        )
        if site:
            stmt = stmt.where(Product.site == site.lower())

        rows = (await self.db.execute(stmt)).all()

        buckets: dict = {}
        for r in rows:
            d = r.day.date() if hasattr(r.day, "date") else r.day
            if d not in buckets:
                buckets[d] = {"InStock": 0, "OutOfStock": 0, "OnOrder": 0}
            key = r.availability or "OutOfStock"
            if key in buckets[d]:
                buckets[d][key] += int(r.count or 0)

        points = [
            AvailabilityTimelinePoint(
                date=d,
                in_stock=v["InStock"],
                out_of_stock=v["OutOfStock"],
                on_order=v["OnOrder"],
            )
            for d, v in sorted(buckets.items())
        ]
        return AvailabilityTimelineResponse(
            site=site, period_days=period_days, points=points
        )

    # ─── Sites overview (NO N+1 — two bulk queries) ───
    async def get_sites_overview(self) -> SitesOverviewResponse:
        # 1. Aggregate products per site
        product_stmt = (
            select(
                Product.site,
                func.count().label("total_products"),
                func.count().filter(Product.is_enabled == True).label("enabled_products"),
                func.count().filter(Product.availability == "InStock").label("in_stock"),
                func.count().filter(Product.availability == "OutOfStock").label("out_of_stock"),
                func.avg(Product.price).label("avg_price"),
            )
            .group_by(Product.site)
            .order_by(Product.site)
        )
        product_rows = (await self.db.execute(product_stmt)).all()

        if not product_rows:
            return SitesOverviewResponse(items=[], total_sites=0)

        sites = [r.site for r in product_rows]

        # 2. Latest batch per site (single query with DISTINCT ON)
        latest_batches: dict[str, MonitoringBatch] = {}
        batch_stmt = (
            select(MonitoringBatch)
            .where(MonitoringBatch.site.in_(sites))
            .order_by(
                MonitoringBatch.site,
                desc(MonitoringBatch.start_time),
            )
            .distinct(MonitoringBatch.site)
        )
        for b in (await self.db.execute(batch_stmt)).scalars().all():
            latest_batches[b.site] = b

        # 3. Unresolved alert counts per site (single query)
        alert_stmt = (
            select(MonitoringAlert.site, func.count().label("cnt"))
            .where(MonitoringAlert.site.in_(sites))
            .where(MonitoringAlert.is_resolved == False)
            .group_by(MonitoringAlert.site)
        )
        alert_counts = {
            row.site: int(row.cnt or 0)
            for row in (await self.db.execute(alert_stmt)).all()
        }

        items = []
        for r in product_rows:
            batch = latest_batches.get(r.site)
            items.append(
                SiteOverview(
                    site=r.site,
                    total_products=int(r.total_products or 0),
                    enabled_products=int(r.enabled_products or 0),
                    in_stock=int(r.in_stock or 0),
                    out_of_stock=int(r.out_of_stock or 0),
                    avg_price=float(r.avg_price or 0),
                    last_batch_status=batch.status if batch else None,
                    last_batch_at=batch.start_time if batch else None,
                    unresolved_alerts=alert_counts.get(r.site, 0),
                )
            )

        return SitesOverviewResponse(items=items, total_sites=len(items))