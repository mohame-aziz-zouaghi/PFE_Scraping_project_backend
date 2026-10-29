# app/services/canonical_verifier.py
"""
LLM verifier for one canonical product group.

Uses Ollama's grammar + a dual system/user prompt structure so the
model reliably outputs our exact schema even when it would
otherwise echo the payload.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import httpx
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging_config import get_logger
from app.models.canonical_product import CanonicalProduct
from app.models.product import Product
from app.services.verification_prompts import CANONICAL_SYSTEM_PROMPT

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Grammar — constrains the output to our schema
# ═══════════════════════════════════════════════════════
_CANONICAL_GRAMMAR = r"""
root    ::= "{" ws "\"decision\"" ws ":" ws decision "," ws
              "\"confidence\"" ws ":" ws confidence "," ws
              "\"reason\"" ws ":" ws string "," ws
              "\"outlier_listing_ids\"" ws ":" ws idarray ws "}"
decision ::= "\"same\"" | "\"split\"" | "\"mismatch\""
confidence ::= "0" | "1" | "0" "." digit+ | "1" "." "0"+
digit   ::= [0-9]
string  ::= "\"" ( [a-zA-Z0-9 \-_.,;:!?'/()#&+=@%]* ) "\""
idarray ::= "[" ws ( string ( ws "," ws string )* )? ws "]"
ws      ::= [ \t\n]*
"""


@dataclass
class CanonicalVerdict:
    decision: str            # same | split | mismatch | error
    confidence: float = 0.0
    reason: str = ""
    outlier_listing_ids: list[UUID] = field(default_factory=list)
    llm_latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_model: str = ""


# ═══════════════════════════════════════════════════════
# Spec normalisation
# ═══════════════════════════════════════════════════════
_COLOUR_MAP = {
    "noir": "black", "noire": "black",
    "blanc": "white", "blanche": "white",
    "rouge": "red",
    "bleu": "blue", "bleue": "blue", "bleu marine": "navy",
    "vert": "green", "verte": "green",
    "gris": "grey", "grise": "grey",
    "argent": "silver",
    "or": "gold",
    "orange": "orange",
    "rose": "pink",
    "violet": "purple",
    "marron": "brown",
}


def _normalise_spec_value(v: str) -> str:
    s = str(v).strip().lower()
    s = re.sub(r"(\d+)\s*(go|gb)\b", r"\1gb", s)
    s = re.sub(r"(\d+)\s*(to|tb)\b", r"\1tb", s)
    s = re.sub(r"(\d+)\s*(?:e|th|ème|ere)\s*gen", r"\1gen", s)
    s = re.sub(r"gen\s*(\d+)", r"\1gen", s)
    s = _COLOUR_MAP.get(s, s)
    return s


def _normalise_specs(specs) -> dict[str, str]:
    """
    Normalise short_description / specs into a flat dict[str, str].

    Handles every shape the scrapers may produce:

        dict                     -> {"couleur": "gris", ...}
        list[str]                -> parse "Key: Value" strings
        list[dict]               -> merge {"key": ..., "value": ...}
                                    or merge every dict directly
        str (JSON)               -> json.loads then recurse
        str (plain "K: V; K: V") -> parse pairs
        None / empty             -> {}

    Any value with a nested structure is JSON-serialised to a short
    string so the LLM still sees it. Never raises — a malformed
    record becomes an empty dict, which downstream treats as
    "no specs known."
    """
    import json

    if specs is None:
        return {}

    # ── str: try JSON, then key:value pairs ────────────────────
    if isinstance(specs, str):
        s = specs.strip()
        if not s:
            return {}
        if s[0] in "{[":
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
            if isinstance(parsed, (dict, list)):
                return _normalise_specs(parsed)
        # Fallback: "Key: Value; Key2: Value2" or "Key: Value"
        return _parse_kv_string(s)

    # ── dict: recurse per value ────────────────────────────────
    if isinstance(specs, dict):
        out: dict[str, str] = {}
        for k, v in specs.items():
            key = str(k).strip().lower()
            if not key:
                continue
            out[key] = _stringify_spec_value(v)
        return out

    # ── list: could be strings or dicts ────────────────────────
    if isinstance(specs, (list, tuple)):
        merged: dict[str, str] = {}
        for item in specs:
            if item is None:
                continue
            if isinstance(item, dict):
                # Case 1: {"key": "Couleur", "value": "Gris"}
                if "key" in item and "value" in item:
                    k = str(item["key"]).strip().lower()
                    if k:
                        merged[k] = _stringify_spec_value(item["value"])
                # Case 2: {"couleur": "gris"} — merge directly
                else:
                    for k, v in item.items():
                        ks = str(k).strip().lower()
                        if ks:
                            merged[ks] = _stringify_spec_value(v)
            elif isinstance(item, str):
                merged.update(_parse_kv_string(item))
            else:
                # Scalar in a list — ignore, can't key it
                continue
        return merged

    # Unknown type — don't raise, just return empty
    return {}


def _parse_kv_string(s: str) -> dict[str, str]:
    """
    Parse "Key: Value; Key2: Value2" or "Key: Value" into a dict.
    If there's no ": " at all, skip the string (can't key it).
    """
    out: dict[str, str] = {}
    # Split on ';' or newlines
    for chunk in s.replace("\n", ";").split(";"):
        if ":" not in chunk:
            continue
        k, _, v = chunk.partition(":")
        key = k.strip().lower()
        if not key:
            continue
        out[key] = v.strip()[:200]
    return out


def _stringify_spec_value(v) -> str:
    """
    Turn any spec value into a short string suitable for the LLM
    prompt. Preserves lists as comma-joined, dicts as JSON, and
    truncates long strings to 200 chars.
    """
    import json

    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()[:200]
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, (list, tuple)):
        parts = [_stringify_spec_value(x) for x in v if x is not None]
        return ", ".join(p for p in parts if p)[:200]
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False)[:200]
        except Exception:
            return ""
    try:
        return str(v)[:200]
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════
# Verifier
# ═══════════════════════════════════════════════════════
class CanonicalVerifier:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.base_url = settings.ollama_base_url
        self.model = getattr(
            settings, "rag_llm_model",
            getattr(settings, "ollama_model", "llama3.1:8b"),
        )

    async def verify(
        self,
        *,
        canonical_id: UUID,
        timeout_s: int = 120,
    ) -> CanonicalVerdict:
        canonical = await self.db.get(CanonicalProduct, canonical_id)
        if canonical is None:
            await self._stamp_verification(
                canonical_id, state="blacklist",
            )
            return CanonicalVerdict(
                decision="error", reason="canonical not found",
            )

        listings = list((await self.db.execute(
            select(Product).where(Product.canonical_id == canonical_id)
        )).scalars().all())

        if len(listings) < 2:
            await self._stamp_verification(
                canonical_id, state="settled",
            )
            return CanonicalVerdict(
                decision="error",
                reason="canonical has fewer than 2 listings",
            )

        payload = self._build_payload(canonical, listings)

        # ── Build a strong user message that includes the
        # instruction so the model doesn't just echo the payload.
        user_message = (
            "DECIDE whether all these listings are the same product.\n"
            "RULES:\n"
            "  - same: every listing is the same model/capacity/colour/gen.\n"
            "  - split: some listings differ, others match.\n"
            "  - mismatch: every listing is different.\n"
            "Return JSON only.\n"
            "\n"
            "LISTINGS:\n"
            + json.dumps(payload, separators=(",", ":"))
        )

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=timeout_s,
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={
                        "model": self.model,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 150,
                            "num_ctx": 2048,
                            "grammar": _CANONICAL_GRAMMAR,
                        },
                        "messages": [
                            {
                                "role": "system",
                                "content": CANONICAL_SYSTEM_PROMPT,
                            },
                            {"role": "user", "content": user_message},
                        ],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            latency = int((time.monotonic() - t0) * 1000)
            reason = f"{type(exc).__name__}: {str(exc)[:180]}"
            logger.warning(
                "canonical_verify_llm_failed",
                canonical_id=str(canonical_id),
                error=reason,
                latency_ms=latency,
            )
            await self._stamp_verification(
                canonical_id, state="pending",
            )
            return CanonicalVerdict(
                decision="error",
                reason=reason,
                llm_latency_ms=latency,
                llm_model=self.model,
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        content = (data.get("message") or {}).get("content", "")
        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))

        verdict = self._parse_verdict(content, listings)
        verdict.llm_latency_ms = latency_ms
        verdict.prompt_tokens = prompt_tokens
        verdict.completion_tokens = completion_tokens
        verdict.llm_model = self.model

        # ── State transition:
        #    same    -> settled (don't re-check)
        #    split   -> in_progress (rematch will fire)
        #    mismatch-> in_progress (rematch will fire)
        #    error   -> pending (retry next cycle)
        if verdict.decision == "same":
            state = "settled"
        elif verdict.decision in ("split", "mismatch"):
            state = "in_progress"
        else:
            state = "pending"

        await self._stamp_verification(
            canonical_id, state=state,
        )

        return verdict

    @staticmethod
    def _build_payload(
        canonical: CanonicalProduct,
        listings: list[Product],
    ) -> dict:
        return {
            "canonical": {
                "name": (canonical.canonical_name or "")[:80],
                "brand": (canonical.canonical_brand or "")[:40],
                "category": (canonical.canonical_category or "")[:40],
            },
            "listings": [
                {
                    "id": str(p.id),
                    "site": p.site,
                    "name": (p.name or "")[:80],
                    "specs": _normalise_specs(p.short_description),
                }
                for p in listings[:10]
            ],
        }

    def _parse_verdict(
        self, raw: str, listings: list[Product],
    ) -> CanonicalVerdict:
        text_body = raw.strip()
        if text_body.startswith("```"):
            text_body = re.sub(r"^```[a-z]*\n?", "", text_body)
            text_body = re.sub(r"\n?```$", "", text_body)

        try:
            data = json.loads(text_body)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text_body, re.DOTALL)
            if not m:
                logger.warning(
                    "canonical_verify_bad_json",
                    raw_prefix=text_body[:200],
                )
                return CanonicalVerdict(
                    decision="error",
                    reason="LLM returned unparseable JSON",
                )
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return CanonicalVerdict(
                    decision="error",
                    reason="LLM returned unparseable JSON",
                )

        if not isinstance(data, dict):
            return CanonicalVerdict(
                decision="error",
                reason="LLM response is not an object",
            )

        # ── Detect payload echo
        if "produits" in data or "products" in data or (
            "listings" in data and "decision" not in data
        ):
            logger.warning(
                "canonical_verify_payload_echo",
                raw_prefix=text_body[:200],
            )
            return CanonicalVerdict(
                decision="error",
                reason="LLM echoed the payload instead of a verdict",
            )

        # v3 uses "decision"; v1 used "same_product"
        decision = str(data.get("decision", "")).lower()
        if decision not in ("same", "split", "mismatch"):
            if "same_product" in data:
                decision = "same" if data["same_product"] else "mismatch"
            else:
                return CanonicalVerdict(
                    decision="error",
                    reason=(
                        f"unknown decision field: {list(data.keys())}"
                    ),
                )

        confidence = float(data.get("confidence", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        reason = str(data.get("reason", ""))[:400]

        real_ids = {p.id for p in listings}
        raw_outliers = data.get("outlier_listing_ids") or []
        outliers: list[UUID] = []
        for x in raw_outliers:
            try:
                uid = UUID(str(x))
            except (ValueError, TypeError):
                continue
            if uid in real_ids:
                outliers.append(uid)

        return CanonicalVerdict(
            decision=decision,
            confidence=confidence,
            reason=reason,
            outlier_listing_ids=outliers,
        )

    async def _stamp_verification(
        self,
        canonical_id: UUID,
        *,
        state: str,
    ) -> None:
        try:
            await self.db.execute(
                text(
                    "UPDATE canonical_products "
                    "SET verified_at = now(), "
                    "    verification_state = :state "
                    "WHERE id = :cid"
                ),
                {"cid": str(canonical_id), "state": state},
            )
            await self.db.commit()
        except Exception as exc:
            logger.warning(
                "canonical_stamp_failed",
                canonical_id=str(canonical_id),
                error=str(exc)[:200],
            )