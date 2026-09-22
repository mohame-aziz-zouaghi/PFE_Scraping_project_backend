"""Alert API contracts."""
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AlertResponse(BaseModel):
    """Single alert as returned by the API."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    product_id: UUID
    site: Optional[str] = None
    alert_type: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    severity: Optional[str] = None
    change_percent: Optional[float] = None
    is_resolved: bool
    resolved_at: Optional[datetime] = None
    created_at: datetime


class PaginatedAlerts(BaseModel):
    """Paginated list of alerts."""
    items: list[AlertResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class AlertResolveResponse(BaseModel):
    """Returned after successfully resolving an alert."""
    message: str
    alert: AlertResponse


class AlertFilter(BaseModel):
    """Query filters for listing alerts (used internally)."""
    site: Optional[str] = Field(default=None, max_length=50)
    alert_type: Optional[str] = Field(default=None, max_length=50)
    severity: Optional[str] = Field(default=None, max_length=20)
    is_resolved: Optional[bool] = False