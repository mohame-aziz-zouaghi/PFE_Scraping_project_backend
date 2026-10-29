# app/services/rag_guardrails.py
"""
Input and output guardrails for the RAG assistant.

Two classes, both pure Python:

    InputGuardrail  — runs BEFORE anything else (cache, retrieval, LLM).
                      Fast regex + heuristic checks. Never calls the LLM.

    OutputGuardrail — runs AFTER generation. Verifies that product names,
                      prices, and URLs in the answer appear in the
                      retrieved context. Strips anything that doesn't.

Neither class touches the database, Redis, or Ollama. They are the
cheapest and most testable layer in the RAG pipeline.

See PHASE_5_RAG_ASSISTANT_GUIDE.md §7 for the design rationale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from app.core.logging_config import get_logger
from app.core.rag_config import (
    RAG,
    RAG_REFUSAL_FALLBACK_CATEGORY,
    RAG_REFUSAL_TEMPLATE,
)
from app.services.rag_retrieval import RetrievedChunk

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════
@dataclass
class InputVerdict:
    """
    The result of `InputGuardrail.check()`.

    Attributes
    ----------
    allowed : bool
        False means the message must be refused immediately
        (with `refusal_reason`) — no cache lookup, no retrieval.
    cleaned : str
        The message after normalization and truncation. This is
        what the pipeline forwards to retrieval.
    refusal_reason : Optional[str]
        Populated when `allowed=False`. One of:
          "empty", "too_long_hard", "injection", "pii",
          "session_invalid", "locale_unsupported"
    pii_kinds : list[str]
        Which PII patterns matched, for logging. Empty on clean.
    """
    allowed: bool
    cleaned: str
    refusal_reason: Optional[str] = None
    pii_kinds: list[str] = field(default_factory=list)


@dataclass
class OutputVerdict:
    """
    The result of `OutputGuardrail.check()`.

    Attributes
    ----------
    answer : str
        The answer after stripping. May be the canonical refusal
        if `too_much_stripped=True`.
    stripped : int
        How many sentences were removed.
    stripped_kinds : list[str]
        Which checks fired: "product_name", "price", "url".
    too_much_stripped : bool
        True if the strip fraction exceeded the threshold.
    is_refusal : bool
        True if the final answer is the canonical refusal.
    """
    answer: str
    stripped: int = 0
    stripped_kinds: list[str] = field(default_factory=list)
    too_much_stripped: bool = False
    is_refusal: bool = False


# ═══════════════════════════════════════════════════════
# Regexes — all bounded, compiled once at import time.
# ═══════════════════════════════════════════════════════

# Prompt-injection keywords. Deliberately broad; the false-
# positive rate on real shopping questions is ~0.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\b", re.I),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"\b(?:jailbreak|jail\s*break)\b", re.I),
    re.compile(r"\bDAN\s+mode\b", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bpretend\s+to\s+be\b", re.I),
    re.compile(r"\breveal\s+(?:your|the)\s+(?:prompt|instructions?)\b", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
    re.compile(r"\boverride\s+(?:your|the)\s+(?:instructions?|rules?)\b", re.I),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior)\b", re.I),
    re.compile(r"<\s*\|.*?\|\s*>"),                 # chat template markers
    re.compile(r"```\s*system\b", re.I),             # fenced system block
    re.compile(r"\[\s*INST\s*\]", re.I),             # llama special tokens
    re.compile(r"<<\s*SYS\s*>>", re.I),
)

# PII detectors. Simple and bounded; not a substitute for a
# real PII library but catches the obvious leaks.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,10}\b"
    ),
    "phone": re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?"
        r"(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\d)"
    ),
    "iban": re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"
    ),
    # Credit cards: 13–19 digits, optional spaces/dashes.
    "card": re.compile(
        r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"
    ),
}

# Intent whitelist. If the normalized message starts with any of
# these prefixes, we skip the "looks like a product name" check
# and treat the input as in-scope. This is the fast path.
_INTENT_PREFIXES: tuple[str, ...] = (
    "how", "what", "which", "where", "when", "who", "why",
    "is", "are", "do", "does", "can", "could", "would", "should",
    "compare", "find", "show", "search", "list", "tell",
    "give", "get", "help",
)

# A short message (< this many words) that starts with a capital
# letter or contains digits is likely a product name (e.g.
# "HP 15-fd0401nk"). These pass through even without an intent
# prefix.
_PRODUCT_LIKE_MAX_WORDS = 8
_PRODUCT_LIKE_MIN_CHARS = 3

# Locales we support. Anything else falls through as English.
_SUPPORTED_LOCALES = {"en", "fr", "ar"}


# ═══════════════════════════════════════════════════════
# Input guardrail
# ═══════════════════════════════════════════════════════
class InputGuardrail:
    """
    Runs first, before cache and before retrieval.

    Checks, in order:
        1. Non-empty after strip.
        2. Hard length cap (RAG.MAX_MESSAGE_CHARS). Longer is
           truncated, not refused — we keep the first N chars.
        3. Prompt injection keywords → refuse.
        4. PII patterns → refuse.
        5. (Soft) intent whitelist OR product-like shape.
           If neither, we still allow — the retrieval relevance
           gate downstream will catch out-of-scope questions.
    """

    def check(
        self,
        *,
        message: Optional[str],
        locale: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> InputVerdict:
        # 1. Empty
        raw = (message or "").strip()
        if not raw:
            return InputVerdict(
                allowed=False, cleaned="", refusal_reason="empty"
            )

        # 2. Hard truncate
        cleaned = raw[: RAG.MAX_MESSAGE_CHARS].strip()

        # 3. Injection
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(cleaned):
                logger.info(
                    "rag_input_rejected_injection",
                    pattern=pattern.pattern[:60],
                )
                return InputVerdict(
                    allowed=False,
                    cleaned=cleaned,
                    refusal_reason="injection",
                )

        # 4. PII
        pii_kinds = self._detect_pii(cleaned)
        if pii_kinds:
            logger.info(
                "rag_input_rejected_pii",
                kinds=pii_kinds,
            )
            return InputVerdict(
                allowed=False,
                cleaned=cleaned,
                refusal_reason="pii",
                pii_kinds=pii_kinds,
            )

        # 5. (soft) intent / product-like shape — informational
        # only, always allowed.
        locale_norm = (locale or "en").lower()
        if locale_norm not in _SUPPORTED_LOCALES:
            # Fall through as English, don't refuse.
            locale_norm = "en"

        return InputVerdict(allowed=True, cleaned=cleaned)

    @staticmethod
    def _detect_pii(message: str) -> list[str]:
        """
        Returns the list of matched PII kinds (empty if clean).
        Deliberately checks only up to the first match per kind
        to keep it fast.
        """
        hits: list[str] = []
        for kind, pattern in _PII_PATTERNS.items():
            if pattern.search(message):
                hits.append(kind)
        return hits

    @staticmethod
    def looks_in_scope(message: str) -> bool:
        """
        True if the message looks like a shopping question. Used
        for logging only — the retrieval relevance gate is the
        real out-of-scope detector.
        """
        s = message.strip().lower()
        if not s:
            return False
        # Intent prefix
        for prefix in _INTENT_PREFIXES:
            if s.startswith(prefix + " ") or s == prefix:
                return True
        # Product-like shape
        words = s.split()
        if len(words) <= _PRODUCT_LIKE_MAX_WORDS and len(s) >= _PRODUCT_LIKE_MIN_CHARS:
            # Has a digit, or looks like a model number
            if any(ch.isdigit() for ch in s):
                return True
            # Or capital-letter token mid-string
            for token in message.split():
                if token[:1].isupper() and len(token) >= 3:
                    return True
        return False


# ═══════════════════════════════════════════════════════
# Output guardrail
# ═══════════════════════════════════════════════════════
class OutputGuardrail:
    """
    Runs after generation. Verifies that every product name,
    price, and URL in the answer appears in the retrieved context.

    The context is a list of `RetrievedChunk`. We build three
    sets once per call:

        known_names     — normalized product / canonical names
        known_prices    — set of floats (rounded to 2dp)
        known_urls      — set of normalized URLs

    Then for each sentence in the answer:

        • If it contains a product name not in known_names → strip.
        • If it contains a price not in known_prices → strip.
        • If it contains a URL not in known_urls → strip.

    Sentences are split on ., !, ? — plus newlines — because the
    LLM tends to use those boundaries. Fragments are kept as-is.
    """

    def check(
        self,
        *,
        answer: str,
        chunks: list[RetrievedChunk],
        is_already_refusal: bool = False,
    ) -> OutputVerdict:
        text = (answer or "").strip()
        if not text:
            return OutputVerdict(
                answer=self._canonical_refusal(chunks),
                is_refusal=True,
            )

        # If the LLM already returned a refusal, normalize and
        # return it. No stripping needed.
        if is_already_refusal or self._looks_like_refusal(text):
            return OutputVerdict(
                answer=self._canonical_refusal(chunks),
                is_refusal=True,
            )

        # ── Build the known sets from the retrieved context.
        known_names, known_prices, known_urls = (
            self._build_known_sets(chunks)
        )

        # ── Split into sentences.
        sentences = self._split_sentences(text)
        if not sentences:
            return OutputVerdict(
                answer=self._canonical_refusal(chunks),
                is_refusal=True,
            )

        kept: list[str] = []
        stripped = 0
        stripped_kinds: list[str] = []

        for sentence in sentences:
            verdict = self._sentence_verdict(
                sentence=sentence,
                known_names=known_names,
                known_prices=known_prices,
                known_urls=known_urls,
            )
            if verdict is None:
                kept.append(sentence)
            else:
                stripped += 1
                stripped_kinds.append(verdict)

        # ── Strip fraction gate.
        total = len(sentences)
        strip_fraction = stripped / total if total else 1.0
        if strip_fraction > RAG.MAX_STRIP_FRACTION:
            logger.info(
                "rag_output_stripped_too_much",
                stripped=stripped,
                total=total,
                kinds=stripped_kinds,
            )
            return OutputVerdict(
                answer=self._canonical_refusal(chunks),
                stripped=stripped,
                stripped_kinds=stripped_kinds,
                too_much_stripped=True,
                is_refusal=True,
            )

        # ── Reassemble.
        out = " ".join(kept).strip()
        if not out:
            return OutputVerdict(
                answer=self._canonical_refusal(chunks),
                stripped=stripped,
                stripped_kinds=stripped_kinds,
                too_much_stripped=True,
                is_refusal=True,
            )

        return OutputVerdict(
            answer=out,
            stripped=stripped,
            stripped_kinds=stripped_kinds,
        )

    # ═══════════════════════════════════════════════════
    # Known-set construction
    # ═══════════════════════════════════════════════════
    def _build_known_sets(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[set[str], set[float], set[str]]:
        names: set[str] = set()
        prices: set[float] = set()
        urls: set[str] = set()

        for c in chunks:
            # Names
            if c.kind == "product":
                n = (c.meta.get("name") or "").strip()
                if n:
                    names.add(self._normalize_name(n))
                    # Add word-level tokens so "HP Victus 15" in
                    # the answer matches "HP Victus 15-fa0031dx".
                    names.update(self._name_tokens(n))
            elif c.kind == "canonical":
                n = (c.meta.get("canonical_name") or "").strip()
                if n:
                    names.add(self._normalize_name(n))
                    names.update(self._name_tokens(n))
            elif c.kind == "help":
                t = (c.meta.get("title") or "").strip()
                if t:
                    names.add(self._normalize_name(t))

            # Prices
            for key in ("price", "discount_price", "cheapest_price"):
                v = c.meta.get(key)
                if v is not None:
                    try:
                        prices.add(round(float(v), 2))
                    except (TypeError, ValueError):
                        pass

            # URLs
            url = c.meta.get("url")
            if url:
                urls.add(self._normalize_url(url))

        return names, prices, urls

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    @staticmethod
    def _name_tokens(name: str) -> set[str]:
        """
        Return 1-word tokens longer than 3 chars and up to
        3-word shingles from `name`. Used so a chunk named
        "HP Victus 15-fa0031dx RTX 4060" also matches the
        answer saying just "HP Victus" or "RTX 4060".
        """
        tokens = re.findall(r"[A-Za-z0-9\-]{3,}", name)
        out: set[str] = {t.lower() for t in tokens}
        # 2-word shingles
        words = name.split()
        for i in range(len(words) - 1):
            shingle = f"{words[i]} {words[i + 1]}".lower()
            if len(shingle) >= 6:
                out.add(shingle)
        return out

    @staticmethod
    def _normalize_url(url: str) -> str:
        return url.strip().lower().split("#", 1)[0]

    # ═══════════════════════════════════════════════════
    # Sentence-level verification
    # ═══════════════════════════════════════════════════
    def _sentence_verdict(
        self,
        *,
        sentence: str,
        known_names: set[str],
        known_prices: set[float],
        known_urls: set[str],
    ) -> Optional[str]:
        """
        Return None if the sentence is clean, or a string
        indicating which kind of check failed.
        """
        s_lower = sentence.lower()

        # ── URL check
        urls_in_sentence = re.findall(
            r"https?://[^\s,;)\]}\"']+", sentence
        )
        for u in urls_in_sentence:
            if self._normalize_url(u) not in known_urls:
                return "url"

        # ── Price check
        # Match "1299 DT", "1 299,00 DT", "1299.00 DT" etc.
        price_hits = re.findall(
            r"(\d[\d\s.,]*)\s*(?:DT|TND|dt|tnd|dinar)",
            sentence,
        )
        for raw in price_hits:
            val = self._parse_price(raw)
            if val is None:
                continue
            if not any(
                abs(val - kp) <= max(1.0, kp * 0.01)
                for kp in known_prices
            ):
                return "price"

        # ── Product-name check
        # Extract candidate product names: sequences of 2–5
        # capitalized tokens with a digit OR an uppercase run.
        for candidate in self._product_name_candidates(sentence):
            c_lower = candidate.lower()
            if c_lower in known_names:
                continue
            if any(c_lower in k or k in c_lower for k in known_names):
                continue
            # Also allow if the candidate's tokens are all
            # substrings of known names.
            tokens = c_lower.split()
            if tokens and all(
                any(tok in k for k in known_names)
                for tok in tokens
            ):
                continue
            return "product_name"

        return None

    @staticmethod
    def _product_name_candidates(sentence: str) -> list[str]:
        """
        Heuristic extraction of product-name-like spans.

        A span qualifies only if it contains a digit — model
        numbers are the one unambiguous signal. This avoids
        flagging capitalized common phrases ("The Best Selling
        Laptop") as product names.
        """
        words = sentence.split()
        candidates: list[str] = []
        i = 0
        while i < len(words):
            j = i
            while j < len(words) and j - i < 6:
                w = words[j].strip(",.;:!?()[]\"'")
                if not w:
                    break
                j += 1
            if j - i >= 2:
                span = " ".join(words[i:j])
                if any(ch.isdigit() for ch in span):
                    candidates.append(
                        span.strip(",.;:!?()[]\"'")
                    )
            i = j if j > i else i + 1
        return candidates

    @staticmethod
    def _parse_price(raw: str) -> Optional[float]:
        """
        Parse "1 299,00" or "1299.00" or "1,299.00" into 1299.00.
        Returns None if it can't be parsed confidently.
        """
        s = raw.strip()
        if not s:
            return None
        # Remove spaces (thousands separators)
        s = s.replace(" ", "").replace("\u00a0", "")
        # If both , and . present, assume the LAST is decimal.
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            # Could be decimal (1299,00) or thousands (1,299)
            # Decide by the digit count after the comma.
            head, _, tail = s.rpartition(",")
            if len(tail) == 2:
                s = head + "." + tail
            else:
                s = s.replace(",", "")
        try:
            return round(float(s), 2)
        except ValueError:
            return None

    # ═══════════════════════════════════════════════════
    # Sentence splitting
    # ═══════════════════════════════════════════════════
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split on sentence terminators and newlines.

        Preserves the terminator. Drops empty fragments.
        Does not attempt to handle abbreviations — the answer
        is at most 3 sentences anyway.
        """
        if not text:
            return []
        pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [p.strip() for p in pieces if p.strip()]

    # ═══════════════════════════════════════════════════
    # Refusal helpers
    # ═══════════════════════════════════════════════════
    @staticmethod
    def _looks_like_refusal(text: str) -> bool:
        """
        True if the LLM's answer already reads as a refusal.
        Matches the canonical string and a couple of close
        variants, since the model sometimes paraphrases.
        """
        s = text.strip().lower()
        if not s:
            return True
        markers = (
            "i don't have enough information",
            "i do not have enough information",
            "i don't have information about that",
            "i do not have information about that",
            "i cannot help with that",
            "i can't help with that",
            "i'm not able to help with that",
            "i am not able to help with that",
        )
        return any(m in s for m in markers)

    @staticmethod
    def _canonical_refusal(
        chunks: list[RetrievedChunk],
    ) -> str:
        """
        The canonical refusal string, with the best available
        category filled in.
        """
        category = RAG_REFUSAL_FALLBACK_CATEGORY
        for c in chunks:
            if c.category:
                category = c.category
                break
        return RAG_REFUSAL_TEMPLATE.format(category=category)