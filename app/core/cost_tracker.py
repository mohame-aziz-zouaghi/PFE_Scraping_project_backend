"""
Cost tracking for LLM calls.

Records every call to `llm_usage` and enforces a global daily budget.
Per-user budgets are enforced at the endpoint layer (Phase 5).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.models.llm_usage import LLMUsage

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Pricing table (USD per 1M tokens)
# ═══════════════════════════════════════════════════════
_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini":            {"input": 0.15, "output": 0.60},
    "gpt-4o":                 {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},
    # Anthropic
    "claude-3-5-haiku":       {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet":      {"input": 3.00, "output": 15.00},
    # Ollama (local — free)
    "llama3.1:8b":            {"input": 0.00, "output": 0.00},
    "nomic-embed-text":       {"input": 0.00, "output": 0.00},
    # Cache marker
    "cache":                  {"input": 0.00, "output": 0.00},
}


def estimate_cost_cents(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Return cost in cents (1 cent = $0.01)."""
    pricing = _PRICING.get(model)
    if not pricing:
        return 0.0
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round((input_cost + output_cost) * 100, 4)


class CostTracker:
    """Records calls and answers budget questions."""

    @staticmethod
    async def record(
        db: AsyncSession,
        *,
        endpoint: str,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: int = 0,
        user_id: Optional[UUID] = None,
        cache_hit: bool = False,
        success: bool = True,
        error_message: Optional[str] = None,
        prompt_version: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> LLMUsage:
        """Insert a usage row. Never raises."""
        total = prompt_tokens + completion_tokens
        cost = (
            0.0
            if cache_hit
            else estimate_cost_cents(model, prompt_tokens, completion_tokens)
        )

        row = LLMUsage(
            user_id=user_id,
            endpoint=endpoint,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_cents=cost,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            success=success,
            error_message=(error_message or "")[:500] or None,
            prompt_version=prompt_version,
            request_id=request_id,
        )
        db.add(row)

        try:
            await db.flush()
        except Exception as e:
            logger.error("cost_tracker_record_failed", error=str(e))

        logger.info(
            "llm_usage_recorded",
            endpoint=endpoint,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_cents=cost,
            cache_hit=cache_hit,
            success=success,
        )
        return row

    @staticmethod
    async def daily_spend_usd(db: AsyncSession) -> float:
        """Sum of cost_cents since UTC midnight, returned as USD."""
        today = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        stmt = select(func.coalesce(func.sum(LLMUsage.cost_cents), 0)).where(
            LLMUsage.created_at >= today,
            LLMUsage.cache_hit.is_(False),
            LLMUsage.success.is_(True),
        )
        total_cents = (await db.execute(stmt)).scalar_one() or 0
        return float(total_cents) / 100.0

    @staticmethod
    async def within_budget(db: AsyncSession) -> bool:
        """True if today's spend is under the configured limit."""
        if settings.ai_daily_cost_limit_usd <= 0:
            return True
        spent = await CostTracker.daily_spend_usd(db)
        if spent >= settings.ai_daily_cost_limit_usd:
            logger.warning(
                "ai_daily_budget_exceeded",
                spent_usd=spent,
                limit_usd=settings.ai_daily_cost_limit_usd,
            )
            return False
        return True
