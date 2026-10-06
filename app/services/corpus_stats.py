"""Corpus statistics used for token weighting."""
from __future__ import annotations

from collections import Counter
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.core.product_tokenizer import tokenize_product
from app.models.product import Product

logger = get_logger(__name__)


async def compute_token_frequencies(
    db: AsyncSession,
    *,
    site: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict[str, float]:
    stmt = select(Product.name, Product.brand, Product.category).where(Product.is_enabled.is_(True))
    if site:
        stmt = stmt.where(Product.site == site.lower())
    if limit is not None:
        stmt = stmt.limit(max(0, limit))

    rows = (await db.execute(stmt)).all()
    counter: Counter[str] = Counter()
    for name, brand, category in rows:
        token_set = tokenize_product(name=name, brand=brand, category=category)
        counter.update(token_set.model_tokens.keys())

    total = max(1, len(rows))
    return {token: count / total for token, count in counter.items()}
