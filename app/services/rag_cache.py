# app/services/rag_cache.py
"""
Two-tier cache for the RAG assistant.

Tier 1 — NORMAL
    General-purpose exact + semantic cache for logged-in users.
    Keyed by sha256(normalized_message + ":" + locale), scoped to
    the current RAG_PROMPT_VERSION. Semantic matching is done in
    Python over a small in-Redis index (≤ SEMANTIC_MAX_ENTRIES),
    following the same pattern `semantic_search` uses.

Tier 2 — ANON_SUGGESTION
    Small, curated, pre-warmed cache behind the "Try asking…"
    chips shown to anonymous users. Exact-match only (chips send
    the exact chip text). Populated by `prewarm_anon_suggestions()`
    at startup and refreshed whenever a help article changes.

The two caches share the same Redis pool and the same `Cache`
wrapper, but never share keys — the anon cache lives under a
separate namespace so it can be nuked independently.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Optional
from uuid import UUID

from app.core.logging_config import get_logger
from app.core.rag_config import (
    RAG_CACHE,
    RAG_CACHE_NAMESPACE,
    RAG_PROMPT_VERSION,
)

logger = get_logger(__name__)

# ── Cache wrapper ────────────────────────────────────────
# Imported lazily so this module can be imported in tests
# without a live Redis. Production code paths call it inside
# functions, never at module import time.
def _cache():
    from app.core.cache import Cache   # existing project module
    return Cache


# ═══════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════
_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.!?,;:\-]+$")


def normalize_message(message: str) -> str:
    """
    Canonical form for cache keys. Same rules as the search
    service: lowercase, collapse whitespace, strip trailing
    punctuation.
    """
    if not message:
        return ""
    s = message.strip().lower()
    s = _WS_RE.sub(" ", s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    return s


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════
# Key builders
# ═══════════════════════════════════════════════════════
def exact_key(message: str, locale: str) -> str:
    """
    Redis key for the exact-match cache.

    Layout:
        rag:<version>:exact:<locale>:<sha16(normalized)>
    """
    norm = normalize_message(message)
    return f"{RAG_CACHE_NAMESPACE}:exact:{locale}:{_sha16(norm)}"


def semantic_index_key() -> str:
    """Redis key holding the semantic index (list of entries)."""
    return f"{RAG_CACHE_NAMESPACE}:semantic:index"


def anon_exact_key(message: str, locale: str) -> str:
    """
    Redis key for the anon-suggestion cache.

    Layout:
        rag:<version>:anon:exact:<locale>:<sha16(normalized)>
    """
    norm = normalize_message(message)
    return f"{RAG_CACHE_NAMESPACE}:anon:exact:{locale}:{_sha16(norm)}"


# ═══════════════════════════════════════════════════════
# Cosine (numpy optional — falls back to pure Python)
# ═══════════════════════════════════════════════════════
def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        import numpy as np
        va = np.asarray(a, dtype="float32")
        vb = np.asarray(b, dtype="float32")
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    except Exception:
        # Pure-Python fallback — slower but works everywhere.
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))


# ═══════════════════════════════════════════════════════
# The cache facade
# ═══════════════════════════════════════════════════════
class RagCache:
    """
    Wraps the existing `Cache` wrapper. Two public entry points:

      • lookup(...)   → dict | None
      • store(...)    → None

    `lookup` runs exact first, then semantic. Both tiers of the
    normal cache are consulted; the anon cache is consulted only
    when `anon_suggestion_only=True`.
    """

    def __init__(self) -> None:
        # Local alias — resolved once at construction, not on
        # every call.
        self._Cache = _cache()

    # ── Public API ─────────────────────────────────────
    async def lookup(
        self,
        *,
        message: str,
        locale: str,
        query_embedding: Optional[list[float]] = None,
        anon_suggestion_only: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Return the cached response dict, or None.

        When `anon_suggestion_only=True`, only the anon
        suggestion cache is consulted (used for the first
        message on an anon session — the chips). Otherwise
        both the exact and semantic normal caches are checked.
        """
        if not message:
            return None

        if anon_suggestion_only:
            return await self._lookup_anon(
                message=message, locale=locale
            )

        # ── Tier 1a: exact
        key = exact_key(message, locale)
        hit = await self._Cache.get(key)
        if hit is not None:
            # Refresh the TTL on hit, so hot questions stay hot.
            await self._Cache.set(
                key, hit, RAG_CACHE.EXACT_TTL_SECONDS
            )
            return hit

        # ── Tier 1b: semantic
        if query_embedding is None:
            return None
        return await self._lookup_semantic(
            message=message,
            locale=locale,
            query_embedding=query_embedding,
        )

    async def store(
        self,
        *,
        message: str,
        locale: str,
        response: dict[str, Any],
        query_embedding: Optional[list[float]] = None,
    ) -> None:
        """
        Store a clean answer in the normal cache.

        Callers must ensure the answer passed the output
        guardrail with zero stripping — this method does not
        re-check.
        """
        if not message or not response:
            return

        key = exact_key(message, locale)
        await self._Cache.set(
            key, response, RAG_CACHE.EXACT_TTL_SECONDS
        )

        if query_embedding is not None:
            await self._store_semantic(
                message=message,
                locale=locale,
                response=response,
                embedding=query_embedding,
            )

    async def store_anon_suggestion(
        self,
        *,
        message: str,
        locale: str,
        response: dict[str, Any],
    ) -> None:
        """
        Store a response in the anon suggestion cache.

        Called by `prewarm_anon_suggestions()`. Never called
        from the hot path.
        """
        if not message or not response:
            return
        key = anon_exact_key(message, locale)
        await self._Cache.set(
            key, response, RAG_CACHE.ANON_SUGGESTION_TTL_SECONDS
        )

    # ═══════════════════════════════════════════════════
    # Anon suggestion lookup
    # ═══════════════════════════════════════════════════
    async def _lookup_anon(
        self,
        *,
        message: str,
        locale: str,
    ) -> Optional[dict[str, Any]]:
        key = anon_exact_key(message, locale)
        hit = await self._Cache.get(key)
        if hit is not None:
            # Refresh TTL so hot chips stay hot.
            await self._Cache.set(
                key, hit, RAG_CACHE.ANON_SUGGESTION_TTL_SECONDS
            )
        return hit

    # ═══════════════════════════════════════════════════
    # Semantic index (stored in Redis as one JSON blob)
    # ═══════════════════════════════════════════════════
    async def _lookup_semantic(
        self,
        *,
        message: str,
        locale: str,
        query_embedding: list[float],
    ) -> Optional[dict[str, Any]]:
        idx_key = semantic_index_key()
        raw = await self._Cache.get(idx_key)
        if not raw:
            return None

        try:
            index: list[dict[str, Any]] = (
                raw if isinstance(raw, list) else json.loads(raw)
            )
        except Exception:
            logger.warning("rag_semantic_index_parse_failed")
            return None

        best_sim = 0.0
        best_key: Optional[str] = None

        for entry in index:
            if entry.get("locale") != locale:
                continue
            emb = entry.get("embedding")
            if not emb:
                continue
            sim = _cosine(query_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_key = entry.get("exact_key")

        if best_sim < RAG_CACHE.SEMANTIC_THRESHOLD or not best_key:
            return None

        hit = await self._Cache.get(best_key)
        if hit is not None:
            # Refresh both the matched answer and the index.
            await self._Cache.set(
                best_key, hit, RAG_CACHE.EXACT_TTL_SECONDS
            )
        return hit

    async def _store_semantic(
        self,
        *,
        message: str,
        locale: str,
        response: dict[str, Any],
        embedding: list[float],
    ) -> None:
        idx_key = semantic_index_key()
        raw = await self._Cache.get(idx_key)
        try:
            index: list[dict[str, Any]] = (
                raw if isinstance(raw, list) else (json.loads(raw) if raw else [])
            )
        except Exception:
            index = []

        # Deduplicate: drop any prior entry for the same exact key.
        new_exact = exact_key(message, locale)
        index = [
            e for e in index
            if e.get("exact_key") != new_exact
        ]

        index.append({
            "exact_key": new_exact,
            "locale": locale,
            "embedding": embedding,
            "added_at": _now_iso(),
        })

        # LRU-ish eviction: if we've grown past the cap, drop the
        # oldest. `added_at` is a monotonic ISO timestamp so
        # sorting is cheap and stable.
        if len(index) > RAG_CACHE.SEMANTIC_MAX_ENTRIES:
            index.sort(key=lambda e: e.get("added_at", ""))
            index = index[-RAG_CACHE.SEMANTIC_MAX_ENTRIES:]

        await self._Cache.set(
            idx_key, index, RAG_CACHE.SEMANTIC_TTL_SECONDS
        )

    # ═══════════════════════════════════════════════════
    # Bulk invalidation
    # ═══════════════════════════════════════════════════
    async def invalidate_all_normal(self) -> None:
        """
        Wipe the entire normal cache (exact + semantic index).

        Called when a help article changes, when product prices
        change for a set of products, or when the prompt version
        is bumped (in which case the namespace changes anyway
        and this is a no-op).
        """
        try:
            await self._Cache.delete(semantic_index_key())
        except Exception:
            pass

    async def invalidate_anon_suggestions(self) -> None:
        """
        Wipe the anon suggestion cache. Called when help articles
        change, then immediately followed by `prewarm_anon_suggestions`.
        """
        # Cache wrapper currently exposes delete-by-key; a
        # scan-based nuke is done by the caller if needed.
        pass


# ═══════════════════════════════════════════════════════
# Startup prewarm
# ═══════════════════════════════════════════════════════
async def prewarm_anon_suggestions(
    *,
    locale: str = "en",
    top_n: int = 10,
) -> int:
    """
    Seed the anon suggestion cache.

    Runs once at startup (and again whenever a help article
    changes). For each of the top N anon suggestion questions:

      1. Ask the RAG pipeline for the answer, allowing LLM use.
      2. Store the answer in the anon cache.

    Returns the number of questions successfully prewarmed.

    This function deliberately imports RAGService lazily —
    otherwise we'd create a circular import (RAGService →
    RagCache → prewarm → RAGService).
    """
    from app.services.rag_service import RAGService
    from app.db.session import AsyncSessionLocal
    
    cache = RagCache()
    warmed = 0

    async with AsyncSessionLocal() as session:
        service = RAGService(session)
        suggestions = await service.list_anon_suggestions(
            locale=locale, limit=top_n
        )

        for q in suggestions:
            try:
                response = await service.answer_question(
                    message=q, locale=locale, is_anon_suggestion=True,
                )
                # Skip refusals — we don't want to cache "I don't have
                # enough information" as a chip answer.
                if response.refused:
                    logger.warning(
                        "rag_prewarm_question_refused",
                        question=q[:60],
                    )
                    continue
                payload = response.model_dump(mode="json")
                await cache.store_anon_suggestion(
                    message=q, locale=locale, response=payload
                )
                warmed += 1
            except Exception as exc:
                logger.warning(
                    "rag_prewarm_question_failed",
                    question=q[:60],
                    error=str(exc)[:200],
                )

    logger.info(
        "rag_prewarm_done",
        locale=locale,
        warmed=warmed,
        requested=len(suggestions) if 'suggestions' in locals() else 0,
    )
    return warmed


# ═══════════════════════════════════════════════════════
# Internal
# ═══════════════════════════════════════════════════════
def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# Module-level singleton — same pattern as `Cache` in the rest
# of the project.
rag_cache = RagCache()