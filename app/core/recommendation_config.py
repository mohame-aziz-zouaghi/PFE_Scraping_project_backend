"""
Centralized tuning for the recommendation engine.

Bump RECS_RERANK_PROMPT_VERSION whenever the LLM prompt changes —
that invalidates the Redis cache automatically.
"""
from __future__ import annotations
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════
# Prompt / cache versioning
# ═══════════════════════════════════════════════════════
RECS_RERANK_PROMPT_VERSION: str = "recs_rerank_v1"
RECS_RERANK_CACHE_NAMESPACE: str = "recs_rerank_v1"

SELLABLE_AVAILABILITY: tuple[str, ...] = (
    "InStock",
    "OnOrder",
)


# ═══════════════════════════════════════════════════════
# Stage 1 — candidate generation
# ═══════════════════════════════════════════════════════
class CandidateThresholds:
    # How many candidates the whole pipeline should work with
    MAX_CANDIDATES: int = 200

    # How many of the user's most recent views to use as seeds
    RECENT_VIEWS_SEED: int = 5

    # How many neighbours per seed
    NEIGHBOURS_PER_SEED: int = 30

    # Category cohort size
    CATEGORY_COHORT_SIZE: int = 20

    # How many of the user's top categories to sample
    TOP_CATEGORIES: int = 3

    # Fallback: newest enabled products
    FALLBACK_SIZE: int = 200

    # ── NEW: search-driven candidates ──────────────────
    # How many of the user's most recent search queries
    # we turn into literal name/category matches.
    RECENT_SEARCHES_TO_MATCH: int = 5
    # Cap on results per query so one broad query doesn't
    # flood the pool.
    SEARCH_COHORT_SIZE: int = 30
    # Only searches inside this window seed candidates.
    # (Hours; default = TIER.TODAY_HOURS.)
    SEARCH_LOOKBACK_HOURS: int = 24


# ═══════════════════════════════════════════════════════
# Stage 2 — ranking
# ═══════════════════════════════════════════════════════
@dataclass(frozen=True)
class RankingConfig:
    # Existing weights (keep as they are)
    CONTENT: float = 0.35
    POPULARITY: float = 0.10
    PRICE_AFFINITY: float = 0.15
    BRAND_AFFINITY: float = 0.15
    CATEGORY_AFFINITY: float = 0.15
    RECENCY: float = 0.10
    SEEN_PENALTY: float = 0.30
    STAGE2_TOP_K: int = 60
    DEFAULT_LIMIT: int = 15
    MAX_LIMIT: int = 50

    # ── Priority boosts (used by the general ranker when a
    #    candidate belongs to a priority band). Applied additively
    #    on top of the weighted signal. Higher = stronger.
    PRIORITY_WISHLIST: float = 3.0
    PRIORITY_SEARCH: float = 2.5
    PRIORITY_CART: float = 2.0
    PRIORITY_CLICK: float = 1.0
    PRIORITY_DWELL: float = 0.0


# ═══════════════════════════════════════════════════════
# Stage 3 — LLM rerank
# ═══════════════════════════════════════════════════════
class LLMRerankThresholds:
    # Concurrency cap for Ollama
    MAX_CONCURRENCY: int = 2

    # Cache TTL for a reranked candidate set
    CACHE_TTL_SECONDS: int = 6 * 3600     # 6 hours

    # Daily LLM call budget (0 = unlimited)
    DAILY_BUDGET: int = 2000

    # Retry count on transient failure
    RETRY_ATTEMPTS: int = 2
    RETRY_JITTER_SECONDS: float = 0.5

    # If a reranked result would reorder a candidate by more than
    # this many positions, we discard the reorder and keep Stage 2
    # ordering. The LLM is asked to respect this too; this is a
    # hard backstop.
    MAX_REORDER: int = 3


# ═══════════════════════════════════════════════════════
# Cold start
# ═══════════════════════════════════════════════════════
COLD_START_MIN_DISTINCT_PRODUCTS: int = 5
COLD_START_MIN_EVENTS: int = 10


# ═══════════════════════════════════════════════════════
# Profile lifecycle
#
# A profile is "fresh" for TTL_HOURS. When it expires:
#   - the next /for-me request rebuilds it synchronously, and
#   - the nightly recs.refresh_all_users task rebuilds it in
#     the background for users who don't visit.
#
# Only events newer than EVENT_WINDOW_DAYS contribute to the
# rebuild, so a user's taste drifts forward with their recent
# behavior instead of being anchored to 180 days of history.
# ═══════════════════════════════════════════════════════
class ProfileThresholds:
    # How long a profile is considered fresh.
    TTL_HOURS: int = 24

    # Only events newer than this contribute to the profile.
    EVENT_WINDOW_DAYS: int = 14

    # Guard against a user hammering /for-me and forcing a rebuild
    # on every request inside a short window.
    REBUILD_COOLDOWN_MINUTES: int = 30

    SEED_HALF_LIFE_DAYS: float = 4.0


# ═══════════════════════════════════════════════════════
# Materialization
# ═══════════════════════════════════════════════════════
class MaterializationThresholds:
    # How long a materialized row stays fresh
    TTL_HOURS: int = 1

    # How many rows to store per user
    PER_USER: int = 30

    # When refreshing, cap the max users in a single run
    MAX_USERS_PER_RUN: int = 5000


# ═══════════════════════════════════════════════════════
# Tier config — time buckets + slot allocation
#
# This is the core of the recency-first behaviour:
#   "today's search + wishlist" > "yesterday's searches" >
#   "clicks / views / dwell" from any earlier time.
#
# Every event contributes
#       base_weight (from EVENT_WEIGHTS)
#     * tier_multiplier(age_hours)
# so a wishlist from 2h ago (5.0 * 3.0 = 15.0) easily beats
# a click from 3 days ago (2.0 * 1.0 = 2.0), and a fresh
# search-driven candidate gets an extra literal-match boost.
# ═══════════════════════════════════════════════════════
@dataclass(frozen=True)
class TierConfig:
    # ── Time buckets (hours since the event happened) ──
    # Fresh: what the user is looking at RIGHT NOW.
    FRESH_HOURS: int = 6
    # Today: what they were interested in earlier today.
    TODAY_HOURS: int = 24
    # Recent: this week's behaviour.
    RECENT_HOURS: int = 24 * 7
    # Old: contributes to long-term taste but should not
    # out-shout fresh signals.
    OLD_HOURS: int = 24 * 30

    # ── Tier multipliers applied on top of EVENT_WEIGHTS ──
    # A wishlist from 2h ago = 5.0 (base) * 3.0 (fresh) = 15.0
    # A wishlist from 3 days ago = 5.0 (base) * 1.0 (recent) = 5.0
    # A wishlist from 3 weeks ago = 5.0 (base) * 0.3 (old) = 1.5
    FRESH_MULT: float = 3.0
    TODAY_MULT: float = 1.8
    RECENT_MULT: float = 1.0
    OLD_MULT: float = 0.3

    # ── Slot allocation for the final feed ──
    # Total = 15. At most RECENT_SLOTS come from today's intent;
    # the rest come from the long-term profile. If today's bucket
    # is empty, the profile fills all 15.
    TOTAL_SLOTS: int = 15
    RECENT_SLOTS: int = 7     # today's intent
    PROFILE_SLOTS: int = 8    # long-term profile
    SEARCH_SLOTS: int = 4     # within RECENT_SLOTS, slots reserved
                              # for direct search-term matches

    # ── Search handling ──
    # A search event has no product_id; we match its query against
    # name/category. This is the boost applied when a candidate
    # literally matches the query string.
    SEARCH_LITERAL_BOOST: float = 2.0
    SEARCH_CATEGORY_BOOST: float = 1.2

    # ── NEW: tier multipliers used when reading back a profile
    #    that was tagged at build time with tier="fresh"/"mixed"/
    #    "long". These mirror the event-time multipliers above
    #    so the two code paths agree.
    PROFILE_TIER_FRESH_MULT: float = 3.0     # tier="fresh"
    PROFILE_TIER_MIXED_MULT: float = 1.8     # tier="mixed"
    PROFILE_TIER_LONG_MULT: float = 1.0      # tier="long"


    BAND_WISHLIST_FRESH: int = 3     # wishlist adds in last 24h
    BAND_SEARCH_FRESH: int   = 3     # search-driven matches (last 24h)
    BAND_CART_FRESH: int     = 2     # add_to_cart in last 24h
    BAND_CLICK_FRESH: int    = 2     # clicks in last 6h
    BAND_DWELL_FRESH: int    = 1     # dwell events in last 6h


# ═══════════════════════════════════════════════════════
# Tier helpers — used by ProfileService, RankingService,
# and CandidateService so the boundaries are defined once.
# ═══════════════════════════════════════════════════════
def tier_multiplier_h(age_hours: float) -> float:
    """Map an event's age (in hours) to its tier multiplier."""
    if age_hours <= TIER.FRESH_HOURS:
        return TIER.FRESH_MULT
    if age_hours <= TIER.TODAY_HOURS:
        return TIER.TODAY_MULT
    if age_hours <= TIER.RECENT_HOURS:
        return TIER.RECENT_MULT
    return TIER.OLD_MULT


def tier_name_h(age_hours: float) -> str:
    """Map an event's age to a tier *name* (for tagging)."""
    if age_hours <= TIER.FRESH_HOURS:
        return "fresh"
    if age_hours <= TIER.TODAY_HOURS:
        return "today"
    if age_hours <= TIER.RECENT_HOURS:
        return "recent"
    return "old"


def profile_tier_multiplier(tier: str) -> float:
    """
    Map a profile-side tier tag ("fresh" / "mixed" / "long")
    to its multiplier. Unknown tags fall back to the long-term
    (neutral) multiplier so old data doesn't break ranking.
    """
    return {
        "fresh": TIER.PROFILE_TIER_FRESH_MULT,
        "mixed": TIER.PROFILE_TIER_MIXED_MULT,
        "long":  TIER.PROFILE_TIER_LONG_MULT,
    }.get(tier, TIER.PROFILE_TIER_LONG_MULT)


# ═══════════════════════════════════════════════════════
# Module-level singletons (unchanged names so imports
# across the codebase keep working).
# ═══════════════════════════════════════════════════════
CANDIDATE = CandidateThresholds()
RANKING = RankingConfig()
LLM_RERANK = LLMRerankThresholds()
PROFILE = ProfileThresholds()
MATERIALIZATION = MaterializationThresholds()
TIER = TierConfig()