# app/services/structured_queries.py
"""
Structured-query layer for the RAG assistant.

Handles questions that are AGGREGATIONS, not semantic lookups:

    "cheapest PC"
    "what is the most expensive laptop"
    "how many HP products do you have"
    "which site has the cheapest RTX 4060"
    "what is the price of the HP Victus"

For these questions the vector retrieval path is wrong:
the answer is one specific row, not "the 8 most similar
products". Vector search has no concept of min/max/count.

This module:
  1. Classifies the message as an aggregation.
  2. Extracts category / brand / price filters.
  3. Runs a targeted SQL query.
  4. Builds a factual answer + source cards from the rows.

It's a pure layer — no LLM. Every answer traces back to a
row in the products table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.core.rag_config import RAG_SELLABLE_AVAILABILITY

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════
@dataclass
class StructuredQueryResult:
    """
    What the structured-query layer returns when it fires.

    `handled=True` means the pipeline should return this
    result as-is (no vector retrieval, no LLM).

    `handled=False` means the message didn't match an
    aggregation intent — fall through to the normal path.
    """
    handled: bool
    answer: str = ""
    source_cards: list[dict] = field(default_factory=list)
    intent: Optional[str] = None
    filters: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════
# Intent detection
# ═══════════════════════════════════════════════════════
# Aggregation keywords. If any of these appears alongside a
# product/category signal, the layer fires.
_CHEAPEST_PATTERNS = (
    re.compile(r"\b(cheapest|less\s+expensive|lowest\s+price|best\s+price)\b", re.I),
    re.compile(r"\bmoins\s+cher\b", re.I),
    re.compile(r"\bpas\s+cher\b", re.I),
    re.compile(r"\bpour\s+moins\s+de\b", re.I),
)
_MOST_EXPENSIVE_PATTERNS = (
    re.compile(r"\b(most\s+expensive|highest\s+price|priciest|most\s+costly)\b", re.I),
    re.compile(r"\bplus\s+cher\b", re.I),
)
_COUNT_PATTERNS = (
    re.compile(r"\bhow\s+many\s+\w+", re.I),
    re.compile(r"\bcombien\s+de\b", re.I),
)
_PRICE_LOOKUP_PATTERNS = (
    re.compile(r"\b(what\s+is\s+the\s+price|how\s+much\s+(?:is|does|cost))", re.I),
    re.compile(r"\bcombien\s+co[ûu]te\b", re.I),
    re.compile(r"\bprix\s+de\b", re.I),
)

# Product / category words that must appear for the
# aggregation to be actionable. If someone asks "what's the
# cheapest" without naming anything, we can't scope the query.
_CATEGORY_WORDS = (
    "pc", "laptop", "ordinateur", "portable",
    "phone", "téléphone", "smartphone", "mobile",
    "tablet", "tablette",
    "monitor", "écran", "ecran",
    "keyboard", "clavier",
    "mouse", "souris",
    "headphone", "casque", "écouteur", "ecouteur",
    "printer", "imprimante",
    "camera", "caméra", "appareil",
    "tv", "télé", "television",
    "gpu", "carte graphique",
    "console", "playstation", "xbox", "switch",
    "watch", "montre",
    "tablet", "tablette",
    "router", "routeur",
    "tv", "television",
    "speaker", "enceinte",
    "charger", "chargeur",
    "cable", "câble",
    "backpack", "sac",
    "trousse", "sac à dos",
    "accessoire", "accessory",
)

# French ↔ English → SQL category LIKE mapping. This is
# how "PC" becomes a real category filter on the products
# table. Add rows as your categories evolve.
_CATEGORY_MAP: dict[str, list[str]] = {
    # PC / laptops
    "pc": ["%pc%", "%portable%", "%ordinateur%", "%laptop%"],
    "laptop": ["%pc%", "%portable%", "%ordinateur%", "%laptop%"],
    "portable": ["%pc%", "%portable%", "%ordinateur%"],
    "ordinateur": ["%pc%", "%portable%", "%ordinateur%"],
    # Phones
    "phone": ["%phone%", "%smartphone%", "%téléphone%", "%telephone%", "%mobile%"],
    "smartphone": ["%phone%", "%smartphone%", "%mobile%"],
    "téléphone": ["%phone%", "%smartphone%", "%téléphone%", "%telephone%"],
    "telephone": ["%phone%", "%smartphone%", "%téléphone%", "%telephone%"],
    # Monitors
    "monitor": ["%monitor%", "%écran%", "%ecran%"],
    "écran": ["%monitor%", "%écran%", "%ecran%"],
    "ecran": ["%monitor%", "%écran%", "%ecran%"],
    # Keyboards
    "keyboard": ["%keyboard%", "%clavier%"],
    "clavier": ["%keyboard%", "%clavier%"],
    # Mice
    "mouse": ["%mouse%", "%souris%"],
    "souris": ["%mouse%", "%souris%"],
    # Headphones
    "headphone": ["%headphone%", "%casque%", "%écouteur%", "%ecouteur%"],
    "casque": ["%headphone%", "%casque%", "%écouteur%", "%ecouteur%"],
    # TVs
    "tv": ["%tv%", "%télévision%", "%television%", "%télé%", "%tele%"],
    "television": ["%tv%", "%télévision%", "%television%"],
    "télé": ["%tv%", "%télévision%", "%television%"],
    # Tablets
    "tablet": ["%tablet%", "%tablette%"],
    "tablette": ["%tablet%", "%tablette%"],
    # Consoles
    "console": ["%console%", "%playstation%", "%xbox%", "%switch%"],
    "playstation": ["%console%", "%playstation%"],
    "xbox": ["%console%", "%xbox%"],
    # Accessories
    "sac": ["%sac%", "%backpack%"],
    "backpack": ["%sac%", "%backpack%"],
    "trousse": ["%trousse%"],
}


# ═══════════════════════════════════════════════════════
# Filter extraction
# ═══════════════════════════════════════════════════════
_PRICE_CEILING_PATTERNS = (
    re.compile(r"\bunder\s+(\d+(?:[.,]\d+)?)\s*(dt|tnd)?\b", re.I),
    re.compile(r"\bbelow\s+(\d+(?:[.,]\d+)?)\s*(dt|tnd)?\b", re.I),
    re.compile(r"\bless\s+than\s+(\d+(?:[.,]\d+)?)\s*(dt|tnd)?\b", re.I),
    re.compile(r"\bmoins\s+de\s+(\d+(?:[.,]\d+)?)\s*(dt|tnd)?\b", re.I),
    re.compile(r"\bpour\s+moins\s+de\s+(\d+(?:[.,]\d+)?)\s*(dt|tnd)?\b", re.I),
)
_PRICE_FLOOR_PATTERNS = (
    re.compile(r"\bover\s+(\d+(?:[.,]\d+)?)\s*(dt|tnd)?\b", re.I),
    re.compile(r"\babove\s+(\d+(?:[.,]\d+)?)\s*(dt|tnd)?\b", re.I),
    re.compile(r"\bmore\s+than\s+(\d+(?:[.,]\d+)?)\s*(dt|tnd)?\b", re.I),
    re.compile(r"\bplus\s+de\s+(\d+(?:[.,]\d+)?)\s*(dt|tnd)?\b", re.I),
)


def _extract_price_filters(message: str) -> tuple[Optional[float], Optional[float]]:
    """Returns (ceiling, floor)."""
    ceiling: Optional[float] = None
    floor: Optional[float] = None
    for pat in _PRICE_CEILING_PATTERNS:
        m = pat.search(message)
        if m:
            try:
                ceiling = float(m.group(1).replace(",", "."))
            except ValueError:
                pass
            break
    for pat in _PRICE_FLOOR_PATTERNS:
        m = pat.search(message)
        if m:
            try:
                floor = float(m.group(1).replace(",", "."))
            except ValueError:
                pass
            break
    return ceiling, floor


def _extract_category_patterns(message: str) -> list[str]:
    """Return the SQL LIKE patterns for the categories mentioned."""
    words = set(w.lower() for w in re.findall(r"\b\w+\b", message))
    patterns: list[str] = []
    for word in words:
        if word in _CATEGORY_MAP:
            for p in _CATEGORY_MAP[word]:
                if p not in patterns:
                    patterns.append(p)
    return patterns


# ═══════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════
class StructuredQueryService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def try_handle(
        self,
        *,
        message: str,
        locale: str = "en",
    ) -> StructuredQueryResult:
        """
        Try to answer the message as a structured aggregation.
        Returns StructuredQueryResult(handled=False) if it's
        not an aggregation intent.
        """
        lower = message.lower()

        # 1. Classify intent
        is_cheapest = any(p.search(message) for p in _CHEAPEST_PATTERNS)
        is_expensive = any(p.search(message) for p in _MOST_EXPENSIVE_PATTERNS)
        is_count = any(p.search(message) for p in _COUNT_PATTERNS)
        is_price_lookup = any(p.search(message) for p in _PRICE_LOOKUP_PATTERNS)

        if not (is_cheapest or is_expensive or is_count or is_price_lookup):
            return StructuredQueryResult(handled=False)

        # 2. Extract filters
        ceiling, floor = _extract_price_filters(message)
        category_patterns = _extract_category_patterns(message)

        # For "cheapest" / "most expensive" queries we NEED a
        # category filter — otherwise "cheapest product" returns
        # a 1.30 DT cardboard gift box, which is useless.
        if (is_cheapest or is_expensive) and not category_patterns:
            # No category mentioned. Can't scope the query.
            # Fall through to vector retrieval.
            return StructuredQueryResult(handled=False)

        # For price lookup on a specific product, we need the
        # product name — but we don't have a name extractor
        # yet, so fall through for now.
        if is_price_lookup and not category_patterns:
            return StructuredQueryResult(handled=False)

        # 3. Pick the query shape
        if is_cheapest:
            return await self._query_extreme(
                order="ASC",
                category_patterns=category_patterns,
                ceiling=ceiling,
                floor=floor,
                intent="cheapest",
                locale=locale,
            )
        if is_expensive:
            return await self._query_extreme(
                order="DESC",
                category_patterns=category_patterns,
                ceiling=ceiling,
                floor=floor,
                intent="most_expensive",
                locale=locale,
            )
        if is_count:
            return await self._query_count(
                category_patterns=category_patterns,
                ceiling=ceiling,
                floor=floor,
                locale=locale,
            )
        if is_price_lookup:
            return await self._query_extreme(
                order="ASC",
                category_patterns=category_patterns,
                ceiling=ceiling,
                floor=floor,
                intent="price_lookup",
                locale=locale,
            )

        return StructuredQueryResult(handled=False)

    # ═══════════════════════════════════════════════════
    # Query: cheapest / most expensive
    # ═══════════════════════════════════════════════════
    async def _query_extreme(
        self,
        *,
        order: str,
        category_patterns: list[str],
        ceiling: Optional[float],
        floor: Optional[float],
        intent: str,
        locale: str,
    ) -> StructuredQueryResult:
        # Build the WHERE clause dynamically.
        where = [
            "p.is_enabled = TRUE",
            "p.availability = ANY(:sellable)",
            "p.price IS NOT NULL",
            "p.price > 0",
        ]
        params: dict = {
            "sellable": list(RAG_SELLABLE_AVAILABILITY),
            "lim": 5,
        }

        # Category filter (any of the patterns matches)
        if category_patterns:
            cat_clauses = []
            for i, pat in enumerate(category_patterns):
                key = f"cat_{i}"
                cat_clauses.append(f"p.category ILIKE :{key}")
                params[key] = pat
            where.append("(" + " OR ".join(cat_clauses) + ")")

        # Effective price = discount_price if positive, else price
        if ceiling is not None:
            where.append(
                "COALESCE(NULLIF(p.discount_price, 0), p.price) "
                "<= :ceiling"
            )
            params["ceiling"] = ceiling
        if floor is not None:
            where.append(
                "COALESCE(NULLIF(p.discount_price, 0), p.price) "
                ">= :floor"
            )
            params["floor"] = floor

        sql = text(f"""
            SELECT
                p.id,
                p.name,
                p.brand,
                p.category,
                p.site,
                p.price,
                p.discount_price,
                p.availability,
                p.main_image,
                p.url,
                COALESCE(NULLIF(p.discount_price, 0), p.price)
                    AS effective_price
            FROM products p
            WHERE {" AND ".join(where)}
            ORDER BY effective_price {order}, p.last_updated_at DESC
            LIMIT :lim
        """)

        try:
            rows = (await self.db.execute(sql, params)).all()
        except Exception as exc:
            logger.warning(
                "structured_query_failed",
                intent=intent,
                error=str(exc)[:200],
            )
            return StructuredQueryResult(handled=False)

        if not rows:
            return StructuredQueryResult(handled=False)

        # Build the answer + cards
        top = rows[0]
        category_label = self._pretty_category(
            top.category, category_patterns
        )

        cards: list[dict] = []
        for r in rows:
            cards.append({
                "kind": "product",
                "product_id": str(r.id),
                "name": r.name or "",
                "site": r.site,
                "price": float(r.effective_price),
                "availability": r.availability,
                "main_image": r.main_image,
                "url": r.url or "",
            })

        # English vs French phrasing
        fr = locale == "fr"
        if intent == "cheapest":
            if fr:
                answer = (
                    f"Le {category_label} le moins cher actuellement "
                    f"en stock est le {top.name} à "
                    f"{float(top.effective_price):.2f} DT "
                    f"(disponible chez {top.site})."
                )
            else:
                answer = (
                    f"The cheapest {category_label} currently in stock "
                    f"is the {top.name} at "
                    f"{float(top.effective_price):.2f} DT "
                    f"(available at {top.site})."
                )
        elif intent == "most_expensive":
            if fr:
                answer = (
                    f"Le {category_label} le plus cher actuellement "
                    f"est le {top.name} à "
                    f"{float(top.effective_price):.2f} DT "
                    f"(chez {top.site})."
                )
            else:
                answer = (
                    f"The most expensive {category_label} currently "
                    f"in stock is the {top.name} at "
                    f"{float(top.effective_price):.2f} DT "
                    f"(at {top.site})."
                )
        else:  # price_lookup
            if fr:
                answer = (
                    f"Le prix actuel est de "
                    f"{float(top.effective_price):.2f} DT "
                    f"chez {top.site}."
                )
            else:
                answer = (
                    f"The current price is "
                    f"{float(top.effective_price):.2f} DT "
                    f"at {top.site}."
                )

        if len(rows) > 1:
            if fr:
                answer += (
                    f"\n\nVoici les {len(rows)} options les plus "
                    f"pertinentes."
                )
            else:
                answer += (
                    f"\n\nHere are the top {len(rows)} matches."
                )

        return StructuredQueryResult(
            handled=True,
            answer=answer,
            source_cards=cards,
            intent=intent,
            filters={
                "category_patterns": category_patterns,
                "ceiling": ceiling,
                "floor": floor,
            },
        )

    # ═══════════════════════════════════════════════════
    # Query: count
    # ═══════════════════════════════════════════════════
    async def _query_count(
        self,
        *,
        category_patterns: list[str],
        ceiling: Optional[float],
        floor: Optional[float],
        locale: str,
    ) -> StructuredQueryResult:
        where = [
            "p.is_enabled = TRUE",
            "p.availability = ANY(:sellable)",
        ]
        params: dict = {"sellable": list(RAG_SELLABLE_AVAILABILITY)}

        if category_patterns:
            cat_clauses = []
            for i, pat in enumerate(category_patterns):
                key = f"cat_{i}"
                cat_clauses.append(f"p.category ILIKE :{key}")
                params[key] = pat
            where.append("(" + " OR ".join(cat_clauses) + ")")

        if ceiling is not None:
            where.append(
                "COALESCE(NULLIF(p.discount_price, 0), p.price) "
                "<= :ceiling"
            )
            params["ceiling"] = ceiling
        if floor is not None:
            where.append(
                "COALESCE(NULLIF(p.discount_price, 0), p.price) "
                ">= :floor"
            )
            params["floor"] = floor

        sql = text(f"""
            SELECT COUNT(*) FROM products p
            WHERE {" AND ".join(where)}
        """)

        try:
            count = int(
                (await self.db.execute(sql, params)).scalar_one()
            )
        except Exception as exc:
            logger.warning(
                "structured_count_failed", error=str(exc)[:200]
            )
            return StructuredQueryResult(handled=False)

        category_label = self._pretty_category(None, category_patterns)
        fr = locale == "fr"

        if fr:
            answer = (
                f"Nous avons {count} produits disponibles dans la "
                f"catégorie {category_label}."
            )
        else:
            answer = (
                f"We have {count} products available in the "
                f"{category_label} category."
            )

        return StructuredQueryResult(
            handled=True,
            answer=answer,
            source_cards=[],
            intent="count",
            filters={"category_patterns": category_patterns},
        )

    # ═══════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════
    @staticmethod
    def _pretty_category(
        raw: Optional[str],
        patterns: list[str],
    ) -> str:
        """Turn a raw category or pattern list into a label."""
        if raw:
            return raw
        if patterns:
            # Use the first pattern, strip the LIKE wildcards
            first = patterns[0].strip("%").lower()
            return first
        return "product"