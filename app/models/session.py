"""Per-product monitoring health metrics — matches the DB schema."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class MonitoringStat(Base):
    __tablename__ = "monitoring_stats"

    __table_args__ = (
        UniqueConstraint("site", "product_id", name="monitoring_stats_site_product_unique"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    product_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=True,
    )

    # Check counts
    check_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    change_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    price_change_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    availability_change_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    # Timing
    last_check_duration: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    avg_check_duration: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # Errors
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    consecutive_errors: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    # Success tracking
    last_success: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<MonitoringStat {self.site}/{self.product_id} "
            f"checks={self.check_count} errors={self.consecutive_errors}>"
        )