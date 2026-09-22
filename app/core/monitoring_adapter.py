"""Optional integration boundary for the monitoring engine."""
from importlib import import_module
from typing import Optional, Protocol, runtime_checkable

from app.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class MonitoringAdapter(Protocol):
    """Interface any external monitoring adapter must implement."""
    def enabled_sites(self) -> list[str]: ...
    def run_site(self, site: str) -> None: ...


class UnavailableMonitoringAdapter:
    """Returned when no adapter is configured — methods fail loudly."""
    reason = (
        "No monitoring adapter configured. "
        "Set MONITORING_ADAPTER in .env as module:factory."
    )

    def enabled_sites(self) -> list[str]:
        return []

    def run_site(self, site: str) -> None:
        raise RuntimeError(self.reason)


_adapter: Optional[MonitoringAdapter] = None
_adapter_loaded: bool = False


def get_monitoring_adapter() -> MonitoringAdapter:
    """
    Get the configured monitoring adapter.

    Loads lazily on first call and caches. Returns
    UnavailableMonitoringAdapter if not configured.
    """
    global _adapter, _adapter_loaded
    if _adapter_loaded:
        return _adapter

    _adapter_loaded = True

    if not settings.monitoring_adapter:
        _adapter = UnavailableMonitoringAdapter()
        return _adapter

    try:
        module_name, sep, factory_name = settings.monitoring_adapter.partition(":")
        if not sep:
            raise RuntimeError(
                "MONITORING_ADAPTER must be 'module:factory', "
                f"got: {settings.monitoring_adapter!r}"
            )
        module = import_module(module_name)
        factory = getattr(module, factory_name)
        _adapter = factory()
        logger.info(
            "monitoring_adapter_loaded",
            module=module_name,
            factory=factory_name,
        )
        return _adapter
    except Exception as e:
        logger.error(
            "monitoring_adapter_load_failed",
            configured=settings.monitoring_adapter,
            error=str(e),
        )
        _adapter = UnavailableMonitoringAdapter()
        return _adapter


def monitoring_status() -> dict:
    """Return current monitoring availability."""
    try:
        adapter = get_monitoring_adapter()
        if isinstance(adapter, UnavailableMonitoringAdapter):
            return {"available": False, "reason": adapter.reason, "sites": []}
        sites = adapter.enabled_sites()
        return {"available": True, "reason": None, "sites": sites}
    except Exception as exc:
        return {"available": False, "reason": str(exc), "sites": []}