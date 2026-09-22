"""
Phase 1 scaffold — creates every file needed for the AI layer foundations.

Usage:
    cd backend
    python -m scripts.scaffold_phase1             # create files
    python -m scripts.scaffold_phase1 --dry-run   # show what would happen
    python -m scripts.scaffold_phase1 --force     # overwrite existing files

What it creates:
    app/core/cache.py
    app/core/cost_tracker.py
    app/core/llm_client.py
    app/core/prompt_templates.py
    app/models/llm_usage.py
    app/repositories/llm_usage_repo.py
    app/services/embedding_service.py
    migrations/005_llm_usage.sql
    scripts/ai_smoke.py
    tests/__init__.py
    tests/core/__init__.py
    tests/core/test_cache.py
    tests/core/test_llm_client.py
    tests/core/test_cost_tracker.py
    tests/core/test_embedding_service.py
    requirements-ai.txt

It does NOT touch existing files unless --force is passed.
It does NOT modify app/config.py — you must add the AI fields manually
(paste from PHASE_1_CONFIG_SNIPPET below into your Settings class).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ─── Locate project root (backend/) ─────────────────────
_THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = _THIS_FILE.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ═══════════════════════════════════════════════════════════════════
# FILE CONTENTS
# ═══════════════════════════════════════════════════════════════════

REQUIREMENTS_AI = """\
# ═══════════════════════════════════════════════════════
# AI layer dependencies — install with:
#   pip install -r requirements-ai.txt
# ═══════════════════════════════════════════════════════
openai>=1.50.0
anthropic>=0.34.0
ollama>=0.3.3

# Time series / ML (used from Phase 4 onward)
numpy>=1.26
scikit-learn>=1.4
statsmodels>=0.14

# Testing
pytest>=8.0
pytest-asyncio>=0.23
respx>=0.21
"""


CACHE_PY = '''\
"""
Redis-backed JSON cache for LLM responses and embeddings.

Every entry is a key + JSON value + TTL. Keys are namespaced so different
features never collide. Fails open — a Redis outage returns None, never raises.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from app.config import settings
from app.core.logging_config import get_logger
from app.db.redis_client import get_redis

logger = get_logger(__name__)


def make_key(namespace: str, *parts: Any) -> str:
    """
    Build a deterministic cache key.

    Example:
        make_key("llm", "recommend", user_id, product_id)
        → "cache:llm:7f3a...b2c1"
    """
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"cache:{namespace}:{digest}"


class Cache:
    """Async JSON cache backed by Redis. Fails open (returns None on error)."""

    @staticmethod
    async def get(key: str) -> Optional[Any]:
        if not settings.ai_cache_enabled:
            return None
        try:
            redis = await get_redis()
            if redis is None:
                return None
            raw = await redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("cache_get_failed", key=key, error=str(e))
            return None

    @staticmethod
    async def set(key: str, value: Any, ttl_seconds: int) -> bool:
        if not settings.ai_cache_enabled:
            return False
        try:
            redis = await get_redis()
            if redis is None:
                return False
            await redis.set(key, json.dumps(value, default=str), ex=ttl_seconds)
            return True
        except Exception as e:
            logger.warning("cache_set_failed", key=key, error=str(e))
            return False

    @staticmethod
    async def delete(key: str) -> None:
        try:
            redis = await get_redis()
            if redis is not None:
                await redis.delete(key)
        except Exception as e:
            logger.warning("cache_delete_failed", key=key, error=str(e))

    @staticmethod
    async def get_or_set(key: str, ttl_seconds: int, producer):
        """
        Return cached value, or call `producer()` and cache the result.

        Usage:
            result = await Cache.get_or_set(key, 3600, lambda: expensive_call())
        """
        cached = await Cache.get(key)
        if cached is not None:
            logger.debug("cache_hit", key=key)
            return cached

        value = await producer()
        if value is not None:
            await Cache.set(key, value, ttl_seconds)
        return value
'''


LLM_USAGE_MODEL_PY = '''\
"""LLM usage — one row per call, tracks cost, tokens, latency."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )

    user_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    endpoint: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    cost_cents: Mapped[float] = mapped_column(
        Numeric(10, 4), default=0, nullable=False
    )
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )

    prompt_version: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        Index("ix_llm_usage_user_created", "user_id", "created_at"),
        Index("ix_llm_usage_endpoint_created", "endpoint", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<LLMUsage {self.endpoint} {self.provider}/{self.model} "
            f"cost={self.cost_cents}c hit={self.cache_hit}>"
        )
'''


COST_TRACKER_PY = '''\
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
'''


LLM_CLIENT_PY = '''\
"""
Provider-agnostic LLM client.

Usage:
    client = LLMClient(db=db, user_id=current_user.id, endpoint="recommend")
    result = await client.chat_json(
        system=SYSTEM_PROMPT,
        user=payload_dict,
        schema=RecommendationVerdict,
        cache_key_parts=["recommend", product_id, user_id],
    )
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional, Type, TypeVar
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.cache import Cache, make_key
from app.core.cost_tracker import CostTracker
from app.core.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when every provider fails."""


class LLMClient:
    def __init__(
        self,
        *,
        db: Optional[AsyncSession] = None,
        user_id: Optional[UUID] = None,
        endpoint: str = "unknown",
        prompt_version: Optional[str] = None,
    ):
        self.db = db
        self.user_id = user_id
        self.endpoint = endpoint
        self.prompt_version = prompt_version

    # ═══════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════
    async def chat_json(
        self,
        *,
        system: str,
        user: dict[str, Any] | str,
        schema: Type[T],
        cache_key_parts: Optional[list[Any]] = None,
        cache_ttl_seconds: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> T:
        """
        Ask the LLM for JSON, validate against `schema`, cache, and record cost.
        Retries once on schema failure. Falls back to secondary provider on
        provider errors.
        """
        # ── 1. Cache check
        cache_key: Optional[str] = None
        if cache_key_parts:
            cache_key = make_key("llm", self.endpoint, *cache_key_parts)
            cached = await Cache.get(cache_key)
            if cached is not None:
                try:
                    parsed = schema.model_validate(cached)
                    if self.db:
                        await CostTracker.record(
                            self.db,
                            endpoint=self.endpoint,
                            provider="cache",
                            model="cache",
                            cache_hit=True,
                            user_id=self.user_id,
                        )
                    logger.debug("llm_cache_hit", endpoint=self.endpoint)
                    return parsed
                except ValidationError:
                    logger.warning(
                        "llm_cache_invalid", endpoint=self.endpoint
                    )

        # ── 2. Budget check
        if self.db and not await CostTracker.within_budget(self.db):
            raise LLMError("Daily AI budget exceeded")

        user_text = (
            user
            if isinstance(user, str)
            else json.dumps(user, default=str, ensure_ascii=False)
        )

        # ── 3. Build attempt list
        attempts: list[tuple[str, str]] = [
            (settings.llm_provider, self._model_for(settings.llm_provider))
        ]
        if settings.llm_fallback_provider:
            attempts.append((
                settings.llm_fallback_provider,
                self._model_for(settings.llm_fallback_provider),
            ))

        last_error: Optional[str] = None
        for provider, model in attempts:
            for retry in range(settings.llm_max_retries + 1):
                strict_system = system
                if retry > 0:
                    strict_system = (
                        system
                        + "\\n\\nSTRICT MODE: Return only valid JSON. "
                        "No prose. No markdown fences. No comments."
                    )

                started = time.monotonic()
                try:
                    raw, usage = await self._call_provider(
                        provider=provider,
                        model=model,
                        system=strict_system,
                        user_text=user_text,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    latency_ms = int((time.monotonic() - started) * 1000)
                    last_error = f"{provider}/{model}: {e}"
                    logger.warning(
                        "llm_call_failed",
                        provider=provider,
                        model=model,
                        retry=retry,
                        error=str(e),
                    )
                    if self.db:
                        await CostTracker.record(
                            self.db,
                            endpoint=self.endpoint,
                            provider=provider,
                            model=model,
                            latency_ms=latency_ms,
                            user_id=self.user_id,
                            success=False,
                            error_message=str(e)[:500],
                            prompt_version=self.prompt_version,
                        )
                    continue

                latency_ms = int((time.monotonic() - started) * 1000)

                # ── 4. Parse
                try:
                    data = self._extract_json(raw)
                    parsed = schema.model_validate(data)
                except (json.JSONDecodeError, ValidationError) as e:
                    last_error = f"schema: {e}"
                    logger.warning(
                        "llm_schema_invalid",
                        provider=provider,
                        model=model,
                        retry=retry,
                        error=str(e)[:300],
                    )
                    if self.db:
                        await CostTracker.record(
                            self.db,
                            endpoint=self.endpoint,
                            provider=provider,
                            model=model,
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            latency_ms=latency_ms,
                            user_id=self.user_id,
                            success=False,
                            error_message=f"schema: {e}"[:500],
                            prompt_version=self.prompt_version,
                        )
                    continue

                # ── 5. Success
                if self.db:
                    await CostTracker.record(
                        self.db,
                        endpoint=self.endpoint,
                        provider=provider,
                        model=model,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        latency_ms=latency_ms,
                        user_id=self.user_id,
                        success=True,
                        prompt_version=self.prompt_version,
                    )

                if cache_key:
                    ttl = cache_ttl_seconds or settings.ai_cache_ttl_hours * 3600
                    await Cache.set(cache_key, parsed.model_dump(), ttl)

                return parsed

        raise LLMError(f"All providers failed. Last error: {last_error}")

    async def chat_text(
        self,
        *,
        system: str,
        user: dict[str, Any] | str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Free-text completion — used by the assistant for streaming later."""
        user_text = (
            user
            if isinstance(user, str)
            else json.dumps(user, default=str, ensure_ascii=False)
        )
        attempts = [
            (settings.llm_provider, self._model_for(settings.llm_provider))
        ]
        if settings.llm_fallback_provider:
            attempts.append((
                settings.llm_fallback_provider,
                self._model_for(settings.llm_fallback_provider),
            ))

        for provider, model in attempts:
            started = time.monotonic()
            try:
                text, usage = await self._call_provider(
                    provider=provider,
                    model=model,
                    system=system,
                    user_text=user_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = int((time.monotonic() - started) * 1000)
                if self.db:
                    await CostTracker.record(
                        self.db,
                        endpoint=self.endpoint,
                        provider=provider,
                        model=model,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        latency_ms=latency_ms,
                        user_id=self.user_id,
                        prompt_version=self.prompt_version,
                    )
                return text
            except Exception as e:
                logger.warning(
                    "llm_text_call_failed", provider=provider, error=str(e)
                )
                continue

        raise LLMError("All providers failed for text completion")

    # ═══════════════════════════════════════════════════════
    # Provider adapters
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _model_for(provider: str) -> str:
        return {
            "openai":    settings.openai_model,
            "ollama":    settings.ollama_model,
            "anthropic": "claude-3-5-haiku",
        }.get(provider, "unknown")

    async def _call_provider(
        self,
        *,
        provider: str,
        model: str,
        system: str,
        user_text: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> tuple[str, dict[str, int]]:
        temp = (
            temperature if temperature is not None else settings.llm_temperature
        )
        tokens = max_tokens or settings.llm_max_tokens

        if provider == "openai":
            return await self._call_openai(model, system, user_text, temp, tokens)
        if provider == "ollama":
            return await self._call_ollama(model, system, user_text, temp, tokens)
        if provider == "anthropic":
            return await self._call_anthropic(
                model, system, user_text, temp, tokens
            )
        raise ValueError(f"Unknown provider: {provider}")

    async def _call_openai(
        self, model: str, system: str, user_text: str,
        temp: float, max_tokens: int,
    ) -> tuple[str, dict[str, int]]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        # json_object mode (not available for all models; skip on failure)
        try:
            kwargs["response_format"] = {"type": "json_object"}
            resp = await client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            resp = await client.chat.completions.create(**kwargs)

        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": (
                resp.usage.completion_tokens if resp.usage else 0
            ),
        }
        return text, usage

    async def _call_ollama(
        self, model: str, system: str, user_text: str,
        temp: float, max_tokens: int,
    ) -> tuple[str, dict[str, int]]:
        import ollama

        client = ollama.AsyncClient(host=settings.ollama_base_url)
        resp = await client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            options={"temperature": temp, "num_predict": max_tokens},
            format="json",
        )
        text = resp["message"]["content"]
        usage = {
            "prompt_tokens": resp.get("prompt_eval_count", 0),
            "completion_tokens": resp.get("eval_count", 0),
        }
        return text, usage

    async def _call_anthropic(
        self, model: str, system: str, user_text: str,
        temp: float, max_tokens: int,
    ) -> tuple[str, dict[str, int]]:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(timeout=settings.llm_timeout_seconds)
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temp,
            system=system,
            messages=[{"role": "user", "content": user_text}],
        )
        text = resp.content[0].text if resp.content else ""
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        }
        return text, usage

    # ═══════════════════════════════════════════════════════
    # JSON extraction
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Robustly pull the first JSON object out of a model's output."""
        raw = raw.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip("` \\n")

        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Find the first {...} block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])

        raise json.JSONDecodeError("No JSON object found", raw, 0)
'''


PROMPT_TEMPLATES_PY = '''\
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
'''


EMBEDDING_SERVICE_PY = '''\
"""
Embedding service — provider-agnostic vector generation.

Text is hashed; identical text is never re-embedded while its cache entry lives.
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.cache import Cache, make_key
from app.core.cost_tracker import CostTracker
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    pass


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingService:
    def __init__(
        self,
        *,
        db: Optional[AsyncSession] = None,
        user_id: Optional[UUID] = None,
        endpoint: str = "embedding",
    ):
        self.db = db
        self.user_id = user_id
        self.endpoint = endpoint

    # ═══════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════
    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Preserves order. Uses cache per item."""
        if not texts:
            return []

        vectors: list[Optional[list[float]]] = [None] * len(texts)
        misses: list[tuple[int, str]] = []

        # ── 1. Cache lookup
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = await Cache.get(key)
            if cached is not None:
                vectors[i] = cached
            else:
                misses.append((i, text))

        if not misses:
            logger.debug("embedding_all_cached", count=len(texts))
            return [v for v in vectors if v is not None]

        # ── 2. Batch call for misses
        batch_size = settings.embedding_batch_size
        for start in range(0, len(misses), batch_size):
            chunk = misses[start : start + batch_size]
            chunk_texts = [t for _, t in chunk]

            vecs, provider, model, usage = await self._embed_provider(chunk_texts)

            # ── 3. Store in cache
            for (i, text), vec in zip(chunk, vecs):
                vectors[i] = vec
                await Cache.set(
                    self._cache_key(text),
                    vec,
                    settings.embedding_cache_ttl_hours * 3600,
                )

            # ── 4. Record cost
            if self.db:
                await CostTracker.record(
                    self.db,
                    endpoint=self.endpoint,
                    provider=provider,
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    latency_ms=0,
                    user_id=self.user_id,
                )

        return [v for v in vectors if v is not None]

    # ═══════════════════════════════════════════════════════
    # Provider calls
    # ═══════════════════════════════════════════════════════
    async def _embed_provider(
        self, texts: list[str]
    ) -> tuple[list[list[float]], str, str, dict]:
        errors: list[str] = []

        if settings.openai_api_key:
            try:
                started = time.monotonic()
                vecs, usage = await self._embed_openai(texts)
                logger.debug(
                    "embedding_openai_ok",
                    count=len(texts),
                    ms=int((time.monotonic() - started) * 1000),
                )
                return vecs, "openai", settings.openai_embedding_model, usage
            except Exception as e:
                errors.append(f"openai: {e}")
                logger.warning("embedding_openai_failed", error=str(e))

        try:
            vecs, usage = await self._embed_ollama(texts)
            return vecs, "ollama", settings.ollama_embedding_model, usage
        except Exception as e:
            errors.append(f"ollama: {e}")
            logger.error("embedding_ollama_failed", error=str(e))

        raise EmbeddingError(
            "All embedding providers failed: " + "; ".join(errors)
        )

    async def _embed_openai(
        self, texts: list[str]
    ) -> tuple[list[list[float]], dict]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )
        resp = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )
        ordered = sorted(resp.data, key=lambda d: d.index)
        vecs = [d.embedding for d in ordered]
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": 0,
        }
        return vecs, usage

    async def _embed_ollama(
        self, texts: list[str]
    ) -> tuple[list[list[float]], dict]:
        import ollama

        client = ollama.AsyncClient(host=settings.ollama_base_url)
        vecs: list[list[float]] = []
        prompt_tokens = 0
        for t in texts:
            resp = await client.embeddings(
                model=settings.ollama_embedding_model, prompt=t
            )
            vecs.append(resp["embedding"])
            prompt_tokens += resp.get("prompt_eval_count", 0)
        return vecs, {"prompt_tokens": prompt_tokens, "completion_tokens": 0}

    # ═══════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _cache_key(text: str) -> str:
        return make_key("embedding", _hash_text(text))
'''


LLM_USAGE_REPO_PY = '''\
"""Read-side queries for llm_usage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.llm_usage import LLMUsage


class LLMUsageRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def summary_by_endpoint(self, *, hours: int = 24) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(
                LLMUsage.endpoint,
                func.count().label("calls"),
                func.sum(LLMUsage.cost_cents).label("cost_cents"),
                func.avg(LLMUsage.latency_ms).label("avg_ms"),
            )
            .where(LLMUsage.created_at >= since)
            .group_by(LLMUsage.endpoint)
            .order_by(func.sum(LLMUsage.cost_cents).desc())
        )
        rows = (await self.db.execute(stmt)).all()
        return [
            {
                "endpoint": r.endpoint,
                "calls": int(r.calls or 0),
                "cost_usd": float(r.cost_cents or 0) / 100.0,
                "avg_latency_ms": float(r.avg_ms or 0),
            }
            for r in rows
        ]

    async def summary_by_user(
        self, *, hours: int = 24, limit: int = 20
    ) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(
                LLMUsage.user_id,
                func.count().label("calls"),
                func.sum(LLMUsage.cost_cents).label("cost_cents"),
            )
            .where(LLMUsage.created_at >= since, LLMUsage.user_id.isnot(None))
            .group_by(LLMUsage.user_id)
            .order_by(func.sum(LLMUsage.cost_cents).desc())
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).all()
        return [
            {
                "user_id": str(r.user_id),
                "calls": int(r.calls or 0),
                "cost_usd": float(r.cost_cents or 0) / 100.0,
            }
            for r in rows
        ]

    async def cache_hit_ratio(self, *, hours: int = 24) -> float:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        total = (
            await self.db.execute(
                select(func.count())
                .select_from(LLMUsage)
                .where(LLMUsage.created_at >= since)
            )
        ).scalar_one() or 0
        hits = (
            await self.db.execute(
                select(func.count())
                .select_from(LLMUsage)
                .where(
                    LLMUsage.created_at >= since,
                    LLMUsage.cache_hit.is_(True),
                )
            )
        ).scalar_one() or 0
        return (hits / total) if total else 0.0
'''


MIGRATION_005 = """\
-- ═══════════════════════════════════════════════════════
-- Migration 005 — LLM usage / cost tracking
-- Safe to run on existing DB (uses IF NOT EXISTS).
-- ═══════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS llm_usage (
    id                 BIGSERIAL PRIMARY KEY,

    user_id            UUID REFERENCES users(id) ON DELETE SET NULL,

    endpoint           VARCHAR(100) NOT NULL,
    provider           VARCHAR(50)  NOT NULL,
    model              VARCHAR(100) NOT NULL,

    prompt_tokens      INTEGER      NOT NULL DEFAULT 0,
    completion_tokens  INTEGER      NOT NULL DEFAULT 0,
    total_tokens       INTEGER      NOT NULL DEFAULT 0,

    cost_cents         NUMERIC(10,4) NOT NULL DEFAULT 0,
    latency_ms         INTEGER      NOT NULL DEFAULT 0,

    cache_hit          BOOLEAN      NOT NULL DEFAULT FALSE,
    success            BOOLEAN      NOT NULL DEFAULT TRUE,
    error_message      VARCHAR(500),

    prompt_version     VARCHAR(20),
    request_id         VARCHAR(64),

    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_llm_usage_user_created
    ON llm_usage (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_llm_usage_endpoint_created
    ON llm_usage (endpoint, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_llm_usage_created_at
    ON llm_usage (created_at DESC);
"""


AI_SMOKE_PY = '''\
"""
Smoke test for Phase 1.

Usage:
    cd backend
    python -m scripts.ai_smoke
"""
import asyncio

from pydantic import BaseModel

from app.core.llm_client import LLMClient
from app.core.prompt_templates import SMOKE_TEST_V1
from app.db.session import async_session_maker
from app.services.embedding_service import EmbeddingService


class EchoResult(BaseModel):
    echo: str
    ok: bool


async def main():
    print("── LLM chat (JSON) ──────────────────────────")
    async with async_session_maker() as db:
        client = LLMClient(
            db=db,
            endpoint="smoke_test",
            prompt_version=SMOKE_TEST_V1.version,
        )
        result = await client.chat_json(
            system=SMOKE_TEST_V1.system,
            user={"echo": "hello"},
            schema=EchoResult,
            cache_key_parts=["smoke", "hello"],
        )
        print(f"  echo={result.echo!r} ok={result.ok}")

        print("\\n── Call again (should hit cache) ────────────")
        result2 = await client.chat_json(
            system=SMOKE_TEST_V1.system,
            user={"echo": "hello"},
            schema=EchoResult,
            cache_key_parts=["smoke", "hello"],
        )
        print(f"  echo={result2.echo!r} ok={result2.ok}")

        print("\\n── Embedding ────────────────────────────────")
        svc = EmbeddingService(db=db, endpoint="smoke_embedding")
        vec = await svc.embed_one("iPhone 15 Pro Max 256GB Titanium")
        print(f"  dimensions: {len(vec)}")
        print(f"  first 3:    {vec[:3]}")

        print("\\n✓ Phase 1 smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
'''


TEST_CACHE_PY = '''\
import pytest

from app.core.cache import Cache, make_key


def test_make_key_deterministic():
    a = make_key("llm", "recommend", 1, 2)
    b = make_key("llm", "recommend", 1, 2)
    assert a == b
    assert a.startswith("cache:llm:")


def test_make_key_differs():
    assert make_key("llm", "a") != make_key("llm", "b")


@pytest.mark.asyncio
async def test_cache_roundtrip():
    key = make_key("test", "roundtrip")
    await Cache.set(key, {"x": 1}, ttl_seconds=10)
    assert await Cache.get(key) == {"x": 1}


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    assert await Cache.get(make_key("test", "nonexistent")) is None


@pytest.mark.asyncio
async def test_get_or_set_calls_producer_once():
    calls = []

    async def producer():
        calls.append(1)
        return {"v": 42}

    key = make_key("test", "gos")
    r1 = await Cache.get_or_set(key, 10, producer)
    r2 = await Cache.get_or_set(key, 10, producer)
    assert r1 == r2 == {"v": 42}
    assert len(calls) == 1
'''


TEST_LLM_CLIENT_PY = '''\
import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from app.core.llm_client import LLMClient, LLMError


class Echo(BaseModel):
    echo: str
    ok: bool


@pytest.mark.asyncio
async def test_chat_json_parses_valid_response():
    client = LLMClient(endpoint="test")
    with patch.object(
        client,
        "_call_provider",
        new=AsyncMock(return_value=(
            json.dumps({"echo": "hi", "ok": True}),
            {"prompt_tokens": 10, "completion_tokens": 5},
        )),
    ):
        result = await client.chat_json(
            system="test", user={"echo": "hi"}, schema=Echo,
        )
    assert result.echo == "hi"
    assert result.ok is True


@pytest.mark.asyncio
async def test_chat_json_extracts_from_markdown_fence():
    client = LLMClient(endpoint="test")
    raw = '```json\\n{"echo": "x", "ok": true}\\n```'
    with patch.object(
        client,
        "_call_provider",
        new=AsyncMock(return_value=(
            raw, {"prompt_tokens": 10, "completion_tokens": 5},
        )),
    ):
        result = await client.chat_json(system="t", user="t", schema=Echo)
    assert result.echo == "x"


@pytest.mark.asyncio
async def test_chat_json_raises_after_all_retries():
    client = LLMClient(endpoint="test")
    with patch.object(
        client,
        "_call_provider",
        new=AsyncMock(return_value=("not json at all", {})),
    ):
        with pytest.raises(LLMError):
            await client.chat_json(system="t", user="t", schema=Echo)


def test_extract_json_handles_prose():
    raw = 'Sure! Here is the answer: {"a": 1} Done.'
    assert LLMClient._extract_json(raw) == {"a": 1}
'''


TEST_COST_TRACKER_PY = '''\
from app.core.cost_tracker import estimate_cost_cents


def test_cost_gpt4o_mini():
    cost = estimate_cost_cents("gpt-4o-mini", 1_000_000, 0)
    assert cost == 15.0


def test_cost_embedding():
    cost = estimate_cost_cents("text-embedding-3-small", 1_000_000, 0)
    assert cost == 2.0


def test_cost_unknown_model_returns_zero():
    assert estimate_cost_cents("nonexistent", 1000, 1000) == 0.0


def test_cost_ollama_is_free():
    assert estimate_cost_cents("llama3.1:8b", 10000, 5000) == 0.0
'''


TEST_EMBEDDING_SERVICE_PY = '''\
from unittest.mock import AsyncMock, patch

import pytest

from app.services.embedding_service import EmbeddingService


@pytest.mark.asyncio
async def test_embed_batch_preserves_order():
    svc = EmbeddingService()
    texts = ["a", "b", "c"]
    fake_vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    with patch.object(
        svc,
        "_embed_provider",
        new=AsyncMock(return_value=(fake_vectors, "openai", "model", {})),
    ):
        result = await svc.embed_batch(texts)
    assert result == fake_vectors


@pytest.mark.asyncio
async def test_embed_batch_uses_cache_on_repeat():
    svc = EmbeddingService()
    texts = ["cached_text"]

    with patch.object(
        svc,
        "_embed_provider",
        new=AsyncMock(return_value=([[1.0, 0.0]], "openai", "model", {})),
    ) as mock_provider:
        await svc.embed_batch(texts)
        await svc.embed_batch(texts)
        assert mock_provider.call_count == 1
'''


CONFIG_SNIPPET = '''\
    # ─── AI / LLM ──────────────────────────────────────
    ai_enabled: bool = Field(default=True)

    llm_provider: str = Field(default="openai")
    llm_fallback_provider: str | None = Field(default="ollama")

    # OpenAI
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_embedding_dims: int = Field(default=1536)

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:8b")
    ollama_embedding_model: str = Field(default="nomic-embed-text")

    # LLM behavior
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=800, ge=1)
    llm_timeout_seconds: int = Field(default=30, ge=1)
    llm_max_retries: int = Field(default=2, ge=0)

    # Cost control
    ai_cache_enabled: bool = Field(default=True)
    ai_cache_ttl_hours: int = Field(default=24, ge=1)
    ai_daily_cost_limit_usd: float = Field(default=1.00, ge=0)
    ai_user_free_credits_per_month: int = Field(default=50, ge=0)

    # Embeddings
    embedding_batch_size: int = Field(default=50, ge=1, le=500)
    embedding_cache_ttl_hours: int = Field(default=168, ge=1)

    @property
    def ai_available(self) -> bool:
        if not self.ai_enabled:
            return False
        return bool(self.openai_api_key) or self.llm_provider == "ollama"
'''


ENV_SNIPPET = """\
# ═══════════════════════════════════════════════════════
# AI / LLM
# ═══════════════════════════════════════════════════════
AI_ENABLED=true

LLM_PROVIDER=openai
LLM_FALLBACK_PROVIDER=ollama

# OpenAI
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMS=1536

# Ollama (local fallback)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# LLM behavior
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=800
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=2

# Cost control
AI_CACHE_ENABLED=true
AI_CACHE_TTL_HOURS=24
AI_DAILY_COST_LIMIT_USD=1.00
AI_USER_FREE_CREDITS_PER_MONTH=50

# Embeddings
EMBEDDING_BATCH_SIZE=50
EMBEDDING_CACHE_TTL_HOURS=168
"""


# ═══════════════════════════════════════════════════════════════════
# FILESYSTEM LAYOUT
# ═══════════════════════════════════════════════════════════════════
FILES: dict[str, str] = {
    "requirements-ai.txt":                              REQUIREMENTS_AI,
    "app/core/cache.py":                                CACHE_PY,
    "app/core/cost_tracker.py":                         COST_TRACKER_PY,
    "app/core/llm_client.py":                           LLM_CLIENT_PY,
    "app/core/prompt_templates.py":                     PROMPT_TEMPLATES_PY,
    "app/models/llm_usage.py":                          LLM_USAGE_MODEL_PY,
    "app/repositories/llm_usage_repo.py":               LLM_USAGE_REPO_PY,
    "app/services/embedding_service.py":                EMBEDDING_SERVICE_PY,
    "migrations/005_llm_usage.sql":                     MIGRATION_005,
    "scripts/ai_smoke.py":                              AI_SMOKE_PY,
    "tests/__init__.py":                                "",
    "tests/core/__init__.py":                           "",
    "tests/core/test_cache.py":                         TEST_CACHE_PY,
    "tests/core/test_llm_client.py":                    TEST_LLM_CLIENT_PY,
    "tests/core/test_cost_tracker.py":                  TEST_COST_TRACKER_PY,
    "tests/core/test_embedding_service.py":             TEST_EMBEDDING_SERVICE_PY,
    "PHASE_1_CONFIG_SNIPPET.py":                        CONFIG_SNIPPET,
    "PHASE_1_ENV_SNIPPET.env":                          ENV_SNIPPET,
}


# ═══════════════════════════════════════════════════════════════════
# SCAFFOLDER
# ═══════════════════════════════════════════════════════════════════
def _ensure_dirs(base: Path, rel_path: str) -> None:
    target = base / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)


def scaffold(*, force: bool, dry_run: bool) -> int:
    created = 0
    skipped = 0

    print(f"Backend root: {BACKEND_ROOT}\n")
    print(f"{'─' * 70}")
    print(f"{'ACTION':<10} FILE")
    print(f"{'─' * 70}")

    for rel_path, content in FILES.items():
        target = BACKEND_ROOT / rel_path
        exists = target.exists()

        if exists and not force:
            print(f"{'skip':<10} {rel_path}  (exists)")
            skipped += 1
            continue

        action = "overwrite" if exists else "create"
        print(f"{action:<10} {rel_path}  ({len(content)} bytes)")

        if dry_run:
            created += 1
            continue

        _ensure_dirs(BACKEND_ROOT, rel_path)
        target.write_text(content, encoding="utf-8")
        created += 1

    print(f"{'─' * 70}")
    print(f"Created: {created}   Skipped: {skipped}   "
          f"Mode: {'DRY-RUN' if dry_run else 'WRITE'}")
    print(f"{'─' * 70}")

    if not dry_run and created > 0:
        print()
        print("NEXT STEPS")
        print("─" * 70)
        print("1. Install AI dependencies:")
        print("     pip install -r requirements-ai.txt")
        print()
        print("2. Add the AI block to app/config.py.")
        print("   Paste from: PHASE_1_CONFIG_SNIPPET.py")
        print("   (insert before the `@property celery_broker_url`)")
        print()
        print("3. Add the AI block to .env.")
        print("   Paste from: PHASE_1_ENV_SNIPPET.env")
        print("   (append to the end of .env)")
        print()
        print("4. Apply the migration:")
        print("     psql -U scraper_aziz -d scraping_db \\")
        print("       -f backend/migrations/005_llm_usage.sql")
        print()
        print("5. Register the model in app/models/__init__.py:")
        print("     from app.models.llm_usage import LLMUsage  # noqa")
        print()
        print("6. Run the tests:")
        print("     pytest tests/core/ -v")
        print()
        print("7. Run the smoke test:")
        print("     python -m scripts.ai_smoke")
        print("─" * 70)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold Phase 1 files.")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files (default: skip).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be created without writing.",
    )
    args = parser.parse_args()
    return scaffold(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())