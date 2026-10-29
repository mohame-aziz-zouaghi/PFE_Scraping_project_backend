# app/services/verification_prompts.py
"""
Prompts for the two Celery verification loops.

Both prompts are:
  - Strict JSON-only, minimal prose
  - Ordered so the model's first token is "{"
  - Token-optimized: input ~80 tokens, output ~100 tokens

Version bumps invalidate the LLM cache automatically.
"""
from __future__ import annotations


CANONICAL_PROMPT_VERSION = "canon_v4"

CANONICAL_SYSTEM_PROMPT = (
    "Reply with JSON only. First character must be {. "
    "You are deciding whether product listings are the same SKU.\n"
    "\n"
    "Decision rules:\n"
    "same if the listings describe the SAME product.\n"
    "split if SOME listings differ on a CRITICAL spec.\n"
    "mismatch if every listing is a different product.\n"
    "\n"
    "CRITICAL specs (any difference → split):\n"
    "  - CPU model (i3 vs i5 vs i7, Ryzen 5 vs 7, etc.)\n"
    "  - GPU model (RTX 4060 vs RTX 4070, Intel UHD vs Iris, etc.)\n"
    "  - RAM capacity (8Go vs 16Go)\n"
    "  - Storage capacity (256Go vs 512Go)\n"
    "  - Connection type (USB vs Bluetooth vs Wireless)\n"
    "  - Screen size (13\" vs 15\" vs 17\")\n"
    "  - Warranty length\n"
    "\n"
    "NON-critical (ignore these differences):\n"
    "  - Marketing wording\n"
    "  - Price (varies by site)\n"
    "  - Stock status\n"
    "  - Operating system (Windows 11 vs Ubuntu — both are\n"
    "    valid OS variants of the same hardware)\n"
    "  - Colour (unless listing explicitly says 'other colour')\n"
    "  - Product name format ('HP Laptop' vs 'Laptop HP')\n"
    "\n"
    "Schema:\n"
    '{"decision":"same"|"split"|"mismatch",'
    '"confidence":0-1,'
    '"reason":"<15 words",'
    '"outlier_listing_ids":["id"]}\n'
    "outlier_listing_ids: only the listings that differ on a "
    "CRITICAL spec. Empty array for same or mismatch."
)


# ═══════════════════════════════════════════════════════
# Recommendation verification
# ═══════════════════════════════════════════════════════
RECOMMENDATION_PROMPT_VERSION = "rec_v3"

RECOMMENDATION_SYSTEM_PROMPT = (
    "Reply with JSON only. First character must be {. "
    "You are deciding whether product recommendations match a "
    "user's intent.\n"
    "\n"
    "A recommendation matches if it shares category, brand, or "
    "keyword with any wishlist item OR any recent search.\n"
    "matches_intent=false if 5 or more recommendations fit nothing.\n"
    "\n"
    "Schema:\n"
    '{"matches_intent":bool,'
    '"confidence":0-1,'
    '"reason":"<15 words",'
    '"mismatched_product_ids":["id"],'
    '"dominant_signal":"wishlist"|"search"|"category"|"mixed"|"unknown"}\n'
    "mismatched_product_ids: only the recommendations that fit "
    "nothing. Empty array if matches_intent=true."
)