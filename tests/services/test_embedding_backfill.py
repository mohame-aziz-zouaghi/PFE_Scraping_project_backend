"""Tests for EmbeddingBackfillService — build_embedding_text is pure, safe to test."""
from unittest.mock import MagicMock

from app.services.embedding_backfill_service import (
    build_embedding_text,
    hash_text,
)


def _make_product(**overrides):
    p = MagicMock()
    p.brand = overrides.get("brand", "ASUS")
    p.name = overrides.get("name", "TUF Gaming F15")
    p.category = overrides.get("category", "Laptops")
    p.short_description = overrides.get("short_description", None)
    return p


def test_build_text_includes_brand_name_category():
    text = build_embedding_text(_make_product())
    assert "ASUS" in text
    assert "TUF Gaming F15" in text
    assert "Laptops" in text
    assert " | " in text


def test_build_text_handles_missing_fields():
    p = _make_product(brand=None, category=None)
    text = build_embedding_text(p)
    assert "TUF Gaming F15" in text
    assert text.count(" | ") >= 0


def test_build_text_truncates_to_max_chars(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "embedding_max_chars", 20, raising=False)
    p = _make_product(name="A" * 100)
    text = build_embedding_text(p)
    assert len(text) <= 20


def test_hash_text_deterministic():
    assert hash_text("hello") == hash_text("hello")
    assert hash_text("hello") != hash_text("world")
