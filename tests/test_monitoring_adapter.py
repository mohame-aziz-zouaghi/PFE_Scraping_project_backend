import pytest

from app.core.monitoring_adapter import UnavailableMonitoringAdapter, monitoring_status


def test_monitoring_is_explicitly_unavailable_without_configuration(monkeypatch):
    monkeypatch.setattr("app.core.monitoring_adapter.settings.monitoring_adapter", "")
    status = monitoring_status()
    assert status["available"] is False
    assert status["sites"] == []
    assert "MONITORING_ADAPTER" in status["reason"]


def test_unavailable_adapter_rejects_execution():
    with pytest.raises(RuntimeError, match="No monitoring adapter"):
        UnavailableMonitoringAdapter().run_site("megapc")
