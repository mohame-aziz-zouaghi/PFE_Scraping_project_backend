"""Product models — mirror the existing PostgreSQL schema."""
from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


# ═══════════════════════════════════════════════════════
# products
# ═══════════════════════════════════════════════════════
class Product(Base):
    __tablename__ = "products"

    # ─── Identity ────────────────────────────────────────
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    site: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    main_product_id: Mapped[Optional[str]] = mapped_column(
        String(100), index=True, nullable=True
    )
    special_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # ─── Descriptive ─────────────────────────────────────
    name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    brand: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # ─── Pricing ─────────────────────────────────────────
    price: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    discount_price: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    discount_amount: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    availability: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )
    price_valid_until: Mapped[Optional[date]] = mapped_column(
        Date, nullable=True
    )

    # ─── Content ─────────────────────────────────────────
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    short_description: Mapped[Optional[Any]] = mapped_column(
        JSONB, nullable=True
    )
    main_image: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    images: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)

    # ─── Lifecycle ───────────────────────────────────────
    scraped_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), nullable=True
    )
    last_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )
    last_checked: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    is_enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=True
    )

    # ─── Monitoring intelligence ─────────────────────────
    monitoring_priority: Mapped[int] = mapped_column(
        Integer, default=3, server_default="3", nullable=True
    )
    monitoring_group: Mapped[str] = mapped_column(
        String(50), default="regular", server_default="regular", nullable=True
    )
    next_check_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    check_failures: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=True
    )
    consecutive_errors: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=True
    )
    total_checks: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=True
    )
    total_errors: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=True
    )

    # ─── Price history stats ─────────────────────────────
    last_price_change: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    price_change_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=True
    )
    price_change_percent: Mapped[Optional[float]] = mapped_column(
        Numeric(6, 2), nullable=True
    )
    price_volatility_score: Mapped[int] = mapped_column(
        Integer, default=50, server_default="50", nullable=True
    )
    avg_price: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 2), nullable=True
    )
    min_price: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 2), nullable=True
    )
    max_price: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 2), nullable=True
    )
    price_trend: Mapped[str] = mapped_column(
        String(20), default="stable", server_default="stable", nullable=True
    )
    change_frequency_7d: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=True
    )
    change_frequency_30d: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=True
    )
    days_since_last_change: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=True
    )

    # ─── Availability history ────────────────────────────
    availability_change_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=True
    )
    last_availability_change: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

        # ── Embedding tracking (added for the RAG pipeline) ──
    embedding_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    last_embedded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    canonical_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("canonical_products.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    def __repr__(self) -> str:
        name = (self.name or "")[:30]
        return f"<Product {self.site}/{self.main_product_id} {name!r}>"


# ═══════════════════════════════════════════════════════
# price_history
# ═══════════════════════════════════════════════════════
class PriceHistory(Base):
    __tablename__ = "price_history"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    product_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    price: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    discount_price: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    discount_amount: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    availability: Mapped[str] = mapped_column(String(20), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        nullable=False,
    )
    change_reason: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )

    def __repr__(self) -> str:
        return f"<PriceHistory {self.product_id} {self.price} @ {self.recorded_at}>"


# ═══════════════════════════════════════════════════════
# product_changes
# ═══════════════════════════════════════════════════════
class ProductChange(Base):
    __tablename__ = "product_changes"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    product_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    change_type: Mapped[str] = mapped_column(
        String(50), index=True, nullable=False
    )
    old_value: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    new_value: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    change_severity: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        nullable=False,
    )
    monitored_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<ProductChange {self.product_id} {self.change_type}>"


# ═══════════════════════════════════════════════════════
# monitoring_alerts  (old_value / new_value are TEXT!)
# ═══════════════════════════════════════════════════════
class MonitoringAlert(Base):
    __tablename__ = "monitoring_alerts"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    product_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    site: Mapped[Optional[str]] = mapped_column(
        String(50), index=True, nullable=True
    )
    alert_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    # NOTE: These are TEXT in your DB, not JSONB
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    severity: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )
    change_percent: Mapped[Optional[float]] = mapped_column(
        Numeric(6, 2), nullable=True
    )
    is_resolved: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<MonitoringAlert {self.id} {self.alert_type} resolved={self.is_resolved}>"


# ═══════════════════════════════════════════════════════
# monitoring_batches
# ═══════════════════════════════════════════════════════
class MonitoringBatch(Base):
    __tablename__ = "monitoring_batches"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    site: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    batch_type: Mapped[str] = mapped_column(
        String(20), default="scheduled", nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20), default="running", nullable=False, index=True
    )
    total_products: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    processed_products: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    changed_products: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    error_products: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_seconds: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )

    def __repr__(self) -> str:
        return f"<MonitoringBatch {self.site} {self.status} ({self.processed_products}/{self.total_products})>"