"""Backward-compatible alert model import.

The canonical definition lives in `app.models.product`. This module
re-exports `MonitoringAlert` so legacy imports keep working.
"""
from app.models.product import MonitoringAlert

__all__ = ["MonitoringAlert"]