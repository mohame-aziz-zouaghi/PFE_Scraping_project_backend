"""Ensure all models are imported for SQLAlchemy metadata."""
from app.models.audit import AuditLog
from app.models.base import Base, TimestampMixin
from app.models.session import MonitoringStat
from app.models.product import (
    MonitoringAlert,
    MonitoringBatch,
    PriceHistory,
    Product,
    ProductChange,
)
from app.models.user import User
from app.models.llm_usage import LLMUsage
from app.models.product_embedding import ProductEmbedding
from app.models.product_match import ProductMatchPair 
from app.models.canonical_product import CanonicalProduct  # noqa
from app.models.base_group import BaseGroup  # noqa


__all__ = [
    "Base",
    "TimestampMixin",
    "User",
    "AuditLog",
    "Product",
    "PriceHistory",
    "ProductChange",
    "MonitoringAlert",
    "MonitoringBatch",
    "MonitoringStat",
    "LLMUsage",
    "ProductEmbedding",
    "ProductMatchPair",
    "CanonicalProduct",
    "BaseGroup",

]