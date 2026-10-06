"""Deterministic product-name tokenization for cross-site matching."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

_KEYWORDS_PATH = Path(__file__).parent.parent / "data" / "matching_keywords.json"


@lru_cache(maxsize=1)
def load_keywords() -> dict:
    with open(_KEYWORDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    t = strip_accents(str(text).lower())
    t = t.replace("’", "'").replace("`", "'")
    t = t.replace("×", "x")
    t = t.replace("″", '"').replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# "S24+" is a different phone from "S24". Splitting on non-word characters
# silently dropped the "+", so both collapsed into the same model token.
# Pack separators such as "SERENA + 6 Chaises" keep their spaces and are untouched.
_PLUS_SUFFIX_RE = re.compile(r"(?<=[a-z0-9])\+(?=\s|$|[,;)/|-])")


def expand_plus(text: str) -> str:
    return _PLUS_SUFFIX_RE.sub(" plus", text)


# ─────────────────────────────────────────────────────────────
# Colours
#
# Colour is a decisive variant attribute (iPhone 15 Noir != iPhone 15 Blanc),
# so the vocabulary must collapse FR/EN spellings and shade names into one
# family, otherwise "Bleu Marine" vs "Awesome Navy" would be a false mismatch.
# The built-in families are merged with the "colors" section of
# matching_keywords.json; the built-in vocabulary wins on conflicts and any JSON
# canonical name that matches a built-in alias is mapped onto its family.
# ─────────────────────────────────────────────────────────────
_COLOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "black": ("noir", "noire", "noirs", "black", "onyx", "jet black"),
    "white": ("blanc", "blanche", "white", "ivoire", "ivory"),
    "gray": ("gris", "grise", "gray", "grey", "graphite", "anthracite"),
    "silver": ("silver", "argent", "argente", "argentee"),
    "gold": ("gold", "dore", "doree"),
    "rose_gold": ("rose gold", "or rose", "gold rose", "rose dore"),
    "pink": ("rose", "pink", "fuchsia"),
    "red": ("rouge", "red"),
    "blue": ("bleu", "bleue", "blue", "navy", "marine"),
    "green": ("vert", "verte", "green", "olive", "kaki", "khaki"),
    "yellow": ("jaune", "yellow"),
    "orange": ("orange",),
    "purple": ("violet", "violette", "purple", "mauve", "lilas", "lavande", "lavender"),
    "brown": ("marron", "brown"),
    "beige": ("beige", "creme", "cream", "sand"),
    "natural": ("naturel", "naturelle", "natural"),
    "clear": ("transparent", "transparente", "translucide", "clear"),
    "wood": ("bois", "wood", "wooden"),
    "inox": ("inox", "stainless steel", "acier inoxydable"),
}


@lru_cache(maxsize=1)
def color_alias_map() -> dict[str, str]:
    own: dict[str, str] = {}
    for family, aliases in _COLOR_FAMILIES.items():
        for alias in aliases:
            own[normalize(alias)] = family

    merged = dict(own)
    for canonical, aliases in load_keywords().get("colors", {}).items():
        canon = normalize(canonical)
        family = own.get(canon, canon)
        for alias in list(aliases) + [canonical]:
            key = normalize(alias)
            if key and key not in own:
                merged[key] = family
    return merged


@lru_cache(maxsize=1)
def _color_regex() -> Optional[re.Pattern]:
    keys = sorted(color_alias_map(), key=len, reverse=True)
    if not keys:
        return None
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![a-z0-9])")


def extract_colors(text: Optional[str]) -> list[str]:
    """Sorted colour families mentioned in the text."""
    rx = _color_regex()
    if rx is None:
        return []
    mapping = color_alias_map()
    return sorted({mapping[m.group(0)] for m in rx.finditer(normalize(text))})


def strip_colors(text: str) -> str:
    rx = _color_regex()
    if rx is None:
        return text
    return re.sub(r"\s+", " ", rx.sub(" ", text)).strip()


# ─────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────
_TOKEN_SPLIT_RE = re.compile(r"[^\w\.]+")
_ALNUM_MODEL_RE = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9\.]+$")
_CAPACITY_RE = re.compile(r"^\d{1,5}(?:gb|tb|go|to)$")
_DIMENSION_RE = re.compile(r"^\d{1,5}(?:mm|cm|in|pouces?|inches?)$")


# Keep the derived tags in the model token set, but do not let them replace exact
# identifiers. Exact hardware tokens are also emitted by variant_signature.py.
_INTEL_CPU_RE = re.compile(r"\bi([3579])\s*[- ]?\s*(\d{4,5})[a-z]*\b", re.I)
_RYZEN_RE = re.compile(r"\bryzen\s*([3579])\s*[- ]?(\d{4})[a-z]*\b", re.I)
_RTX_RE = re.compile(r"\brtx\s*[- ]?(\d{4})(?:\s*(ti|super))?\b", re.I)
_APPLE_M_RE = re.compile(r"\bm(\d)\s*(pro|max|ultra)?\b", re.I)


def expand_hardware_tokens(text: str) -> set[str]:
    t = normalize(text)
    if not t:
        return set()

    tags: set[str] = set()

    for m in _INTEL_CPU_RE.finditer(t):
        tags.update({"x:cpu", "x:intel", f"x:i{m.group(1)}"})
        digits = m.group(2)
        if len(digits) >= 4:
            tags.add(f"x:gen{digits[:2]}")
        tags.add(f"x:cpu_model:i{m.group(2)}")

    for m in _RYZEN_RE.finditer(t):
        tags.update({"x:cpu", "x:amd", f"x:ryzen{m.group(1)}"})
        tags.add(f"x:cpu_model:ryzen{m.group(2)}")
        tags.add(f"x:ryzen_gen{m.group(2)[0]}")

    for m in _RTX_RE.finditer(t):
        suffix = f"_{m.group(2).lower()}" if m.group(2) else ""
        tags.update({"x:gpu", "x:nvidia", f"x:rtx{m.group(1)[:2]}"})
        tags.add(f"x:gpu_model:rtx{m.group(1)}{suffix}")

    for m in _APPLE_M_RE.finditer(t):
        base = f"m{m.group(1)}"
        tags.update({"x:apple", "x:chip", f"x:{base}"})
        if m.group(2):
            tags.add(f"x:{base}_{m.group(2).lower()}")

    return tags


@dataclass
class TokenSet:
    modifiers: dict[str, str] = field(default_factory=dict)
    brands: set[str] = field(default_factory=set)
    model_tokens: dict[str, float] = field(default_factory=dict)

    def all_model_tokens(self) -> set[str]:
        return set(self.model_tokens)


def _normalise_unit_tokens(text: str) -> str:
    """Normalize common French ecommerce unit spellings before tokenization."""
    replacements = [
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:go|g[oô])\b", r"\1gb"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:to|t[oô])\b", r"\1tb"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:gigaoctets?)\b", r"\1gb"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:teraoctets?)\b", r"\1tb"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:metres?|mètres?|meter|meters)\b", r"\1m"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:centimetres?|centimeters?)\b", r"\1cm"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:millimetres?|millimeters?)\b", r"\1mm"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:watts?|watt)\b", r"\1w"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:kg|kgs|kilogrammes?|kilograms?)\b", r"\1kg"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:tr|trs|tours?|rpm)\b", r"\1rpm"),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:mhz)\b", r"\1mhz"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text


def tokenize_product(*, name: Optional[str], brand: Optional[str], category: Optional[str]) -> TokenSet:
    kw = load_keywords()
    stopwords = {normalize(s) for s in kw.get("stopwords", [])}
    spec_keywords = {normalize(s) for s in kw.get("spec_keywords", [])}
    sizes = {normalize(s) for s in kw.get("sizes", [])}
    boosts = kw.get("weight_boosts", {})

    text = _normalise_unit_tokens(expand_plus(normalize(name or "")))
    colors = extract_colors(text)
    text = strip_colors(text)
    text = re.sub(r"\b\d{1,2}\s*(?:eme|ème|e|th)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    raw_tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]

    modifiers: dict[str, str] = {}
    if colors:
        modifiers["color"] = colors[0]
    for token in raw_tokens:
        if _CAPACITY_RE.match(token):
            modifiers.setdefault("capacity", token)
        elif _DIMENSION_RE.match(token):
            modifiers.setdefault("dimension", token)
        elif token in sizes:
            modifiers.setdefault("size", token)

    brands: set[str] = set()
    if brand:
        for token in _TOKEN_SPLIT_RE.split(normalize(brand)):
            if token and token not in stopwords:
                brands.add(token)

    model: dict[str, float] = {}
    for token in raw_tokens:
        if token in stopwords or token in brands:
            continue
        if _CAPACITY_RE.match(token) or _DIMENSION_RE.match(token) or token in sizes:
            continue
        if re.fullmatch(r"\d+(?:mhz|ghz|w|rpm|kg)", token):
            continue

        if _ALNUM_MODEL_RE.match(token):
            weight = boosts.get("alphanumeric", 3.0)
        elif len(token) >= 6:
            weight = boosts.get("long_word", 1.5)
        else:
            weight = boosts.get("common_word", 1.0)
        if token in spec_keywords:
            weight += boosts.get("spec_keyword_bonus", 0.5)
        model[token] = max(model.get(token, 0.0), float(weight))

    for tag in expand_hardware_tokens(f"{name or ''} {category or ''}"):
        model[tag] = max(model.get(tag, 0.0), 2.0)

    return TokenSet(modifiers=modifiers, brands=brands, model_tokens=model)