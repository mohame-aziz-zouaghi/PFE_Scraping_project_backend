"""
Centralized tuning parameters for the cross-site matching pipeline.

Single source of truth for every threshold, prompt version, and
kill-switch. Bumping MATCH_VERIFIER_PROMPT_VERSION invalidates the
LLM cache automatically (Gap 7).
"""
from __future__ import annotations


# ═══════════════════════════════════════════════════════
# Prompt versioning — bump to invalidate the LLM cache.
# ═══════════════════════════════════════════════════════
MATCH_VERIFIER_PROMPT_VERSION: str = "match_verifier_v4"
MATCH_VERIFIER_CACHE_NAMESPACE: str = "match_verify_v4"


# ═══════════════════════════════════════════════════════
# Variant (SKU) level
# ═══════════════════════════════════════════════════════
class MatchingThresholds:
    # Embedding distance bands
    AUTO_ACCEPT_DISTANCE: float = 0.15
    LLM_VERIFY_MAX_DISTANCE: float = 0.25

    # Lexical-confidence bands
    MIN_EDGE_CONFIDENCE: float = 0.70

    # Price sanity gate
    MAX_PRICE_RATIO: float = 2.00

    # LLM concurrency + retries (Gap 3, Gap 4)
    LLM_MAX_CONCURRENCY: int = 2
    LLM_RETRY_ATTEMPTS: int = 2
    LLM_RETRY_JITTER_SECONDS: float = 0.5


# ═══════════════════════════════════════════════════════
# Base-group level
# ═══════════════════════════════════════════════════════
class BaseThresholds:
    # Base grouping is now deterministic (base_signature equality), so
    # no distance threshold is needed. Kept as a placeholder for future
    # fuzzy-base experiments.
    MIN_VARIANTS_PER_BASE: int = 1
    MIN_SITES_PER_BASE: int = 2


VARIANT = MatchingThresholds()
BASE = BaseThresholds()