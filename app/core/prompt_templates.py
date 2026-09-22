"""
Versioned prompt registry.

Every prompt has an ID like "match_verifier_v1" and a build() function
that returns the final system+user strings.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    description: str


# ═══════════════════════════════════════════════════════
# Phase 1 — smoke test prompt (used in unit tests only)
# ═══════════════════════════════════════════════════════
SMOKE_TEST_V1 = Prompt(
    version="smoke_test_v1",
    description="Trivial JSON response used to verify the LLM client works.",
    system=(
        "You are a test assistant. The user will send you a JSON object "
        "with a field called 'echo'. Reply with a JSON object of the form "
        '{"echo": "<the same value>", "ok": true}. Reply with JSON only.'
    ),
)


# ═══════════════════════════════════════════════════════
# Placeholders for later phases
# ═══════════════════════════════════════════════════════
# MATCH_VERIFIER_V1  = Prompt(...)   # Phase 3
# RECOMMENDATION_V1  = Prompt(...)   # Phase 5
# ASSISTANT_V1       = Prompt(...)   # Phase 6


def get_prompt(name: str) -> Prompt:
    """Look up by name; raise if missing (fail fast in dev)."""
    registry = {p.version: p for p in (SMOKE_TEST_V1,)}
    if name not in registry:
        raise KeyError(f"Unknown prompt: {name}")
    return registry[name]
