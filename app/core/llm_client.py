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
                        + "\n\nSTRICT MODE: Return only valid JSON. "
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
            raw = raw.strip("` \n")

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
