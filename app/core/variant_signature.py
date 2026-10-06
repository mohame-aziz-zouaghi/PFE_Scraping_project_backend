"""Deterministic variant/identity extraction for cross-site product matching.

The matcher treats exact variant-defining specifications as hard constraints while
allowing harmless listing differences such as colour and screen-size wording.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Optional

from app.core.product_tokenizer import load_keywords, normalize, strip_accents

_CAPACITY_RE = re.compile(r"\b(\d{1,5})\s*(gb|tb|go|to|l|litres?|liters?)\b", re.I)
_SLASH_CAPACITY_RE = re.compile(r"\b(\d{1,5})\s*/\s*(\d{1,5})\s*(gb|tb|go|to)\b", re.I)
_SCREEN_RE = re.compile(r"\b(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:\"|''|pouces?|inch(?:es)?)\b", re.I)
_LENGTH_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(mm|cm|m|metre|metres|meter|meters|metre?s?)\b", re.I)
_WATT_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*w(?:att|atts)?\b", re.I)
_SPEED_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(mhz|ghz)\b", re.I)
_RPM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*rpm\b", re.I)
_CHAIR_RE = re.compile(r"\b(\d+)\s*(?:chairs?|cha[iî]ses?|places?)\b", re.I)
_UNIT_REPLACEMENTS = (
    (re.compile(r"(?i)(\d+(?:[.,]\d+)?)\s*(?:metres?|mètres?|meter|meters)\b"), r"\1m"),
    (re.compile(r"(?i)(\d+(?:[.,]\d+)?)\s*(?:kg|kgs|kilogrammes?|kilograms?)\b"), r"\1kg"),
    (re.compile(r"(?i)(\d+(?:[.,]\d+)?)\s*(?:tr|trs|tours?|rpm)\b"), r"\1rpm"),
    (re.compile(r"(?i)(\d+(?:[.,]\d+)?)\s*(?:watts?|watt)\b"), r"\1w"),
    (re.compile(r"(?i)(\d+(?:[.,]\d+)?)\s*(?:mhz|ghz)\b"), r"\1mhz"),
)

_MODEL_SUFFIXES = {"pro", "max", "plus", "ultra", "mini", "lite", "fe", "se", "air", "edge"}
_MODEL_PREFIXES = {
    "iphone", "ipad", "galaxy", "redmi", "poco", "pixel", "mate", "nova",
    "oneplus", "xperia", "thinkpad", "ideapad", "yoga", "loq", "vivobook",
    "zenbook", "rog", "tuf", "macbook", "imac", "surface", "ipad", "tab",
    "pad", "watch", "wh", "wf", "buds", "note", "rtx", "gtx", "geforce",
}
_GENERIC_STOP = {
    "new", "nouveau", "original", "officiel", "official", "avec", "with", "sans",
    "plus", "pack", "bundle", "gaming", "gamer", "portable", "ordinateur", "pc",
    "laptop", "notebook", "smartphone", "phone", "telephone", "tablette", "tablet",
    "gris", "gray", "grey", "noir", "black", "blanc", "white", "rouge", "red",
    "bleu", "blue", "vert", "green", "naturel", "natural", "titane", "titanium",
    "ssd", "hdd", "ram", "rom", "wifi", "bluetooth", "4g", "5g", "8k", "uhd",
    "machine", "laver", "lave", "linge", "frontale", "chargeur", "rapide",
    "cable", "câble", "hdmi", "usb", "type", "vers", "ports", "port",
    "carte", "graphique", "mere", "mère", "watercooling", "composants",
    "informatique", "ecouteurs", "sans", "fil", "wireless", "true",
}


def _description_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v)
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value)


def _normalize_spec_text(text: str) -> str:
    text = normalize(text)
    for pattern, repl in _UNIT_REPLACEMENTS:
        text = pattern.sub(repl, text)
    return text


def _product_text_parts(product: Any) -> tuple[str, str, str]:
    name = _normalize_spec_text(_description_to_text(getattr(product, "name", None)))
    short = _normalize_spec_text(_description_to_text(getattr(product, "short_description", None)))
    return name, short, f"{name} {short}".strip()


def _combined_text(product: Any) -> str:
    return _product_text_parts(product)[2]


def _canonical_capacity(value: str, unit: str) -> str:
    n = value.replace(",", ".")
    unit = unit.lower()
    if unit in {"gb", "go"}:
        return f"{n}gb"
    if unit in {"tb", "to"}:
        return f"{n}tb"
    return f"{n}l"


def _extract_capacities(text: str) -> list[str]:
    values: list[str] = []
    slash = list(_SLASH_CAPACITY_RE.finditer(text))
    slash_spans = {m.span() for m in slash}
    for m in slash:
        # 6/128GB is common in tablet/phone listings: treat both as capacities.
        values.append(f"{m.group(1)}gb")
        values.append(_canonical_capacity(m.group(2), m.group(3)))
    for m in _CAPACITY_RE.finditer(text):
        if any(start <= m.start() < end for start, end in slash_spans):
            continue
        values.append(_canonical_capacity(m.group(1), m.group(2)))
    return sorted(set(values), key=lambda x: (float(re.match(r"[\d.]+", x).group()), x))


def _extract_screen(text: str) -> list[str]:
    out = []
    for m in _SCREEN_RE.finditer(text):
        out.append(f"{float(m.group(1).replace(',', '.')):.1f}in")
    return sorted(set(out))


def _extract_hard_specs(text: str) -> dict[str, list[str]]:
    lengths: list[str] = []
    for m in _LENGTH_RE.finditer(text):
        n = float(m.group(1).replace(',', '.'))
        unit = m.group(2).lower()
        if unit.startswith("cm"):
            n /= 100
        elif unit.startswith("mm"):
            n /= 1000
        lengths.append(f"{n:g}m")

    watts = [f"{float(m.group(1).replace(',', '.')):g}w" for m in _WATT_RE.finditer(text)]
    speeds = [f"{float(m.group(1).replace(',', '.')):g}{m.group(2).lower()}" for m in _SPEED_RE.finditer(text)]
    rpms = [f"{float(m.group(1).replace(',', '.')):g}rpm" for m in _RPM_RE.finditer(text)]
    chairs = [f"{m.group(1)}chairs" for m in _CHAIR_RE.finditer(text)]
    return {
        "length": sorted(set(lengths)),
        "wattage": sorted(set(watts)),
        "speed": sorted(set(speeds)),
        "rpm": sorted(set(rpms)),
        "chair_count": sorted(set(chairs)),
    }


def _color_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in load_keywords().get("colors", {}).items():
        for alias in aliases:
            lookup[normalize(alias)] = normalize(canonical)
    return lookup


def _extract_color(text: str) -> Optional[str]:
    lookup = _color_lookup()
    for alias in sorted(lookup, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
            return lookup[alias]
    return None


def _model_tokens(text: str, brand: Optional[str]) -> list[str]:
    text = normalize(text)
    tokens = [t for t in re.split(r"[^a-z0-9.]+", text) if t]
    brand_tokens = set(re.split(r"[^a-z0-9]+", normalize(brand or "")))
    colors = set(_color_lookup())
    kw = load_keywords()
    stop = set(_GENERIC_STOP) | {normalize(x) for x in kw.get("stopwords", [])}

    filtered: list[str] = []
    for i, t in enumerate(tokens):
        if t in brand_tokens or t in colors or t in stop:
            continue
        if re.fullmatch(r"\d+(?:gb|tb|go|to|mhz|ghz|rpm|w|kg|m|cm|mm)", t):
            continue
        if re.fullmatch(r"x\d+", t):
            continue
        if t.isdigit() and (i == 0 or tokens[i - 1] not in _MODEL_PREFIXES):
            continue
        filtered.append(t)

    if not filtered:
        return []

    digit_indices = [i for i, t in enumerate(filtered) if any(c.isdigit() for c in t)]
    if digit_indices:
        # Standalone numbers (e.g. 400 litres, 9 kg, 3 ports) are not model
        # identifiers. A pure numeric token becomes part of a model only when
        # preceded by a known family token such as iPhone, Galaxy, Note, etc.
        for idx in digit_indices:
            token = filtered[idx]
            if token.isdigit() and (idx == 0 or filtered[idx - 1] not in _MODEL_PREFIXES):
                continue
            chosen: list[str] = []
            if idx > 0 and filtered[idx - 1] in _MODEL_PREFIXES:
                chosen.append(filtered[idx - 1])
            chosen.append(token)
            j = idx + 1
            while j < len(filtered) and filtered[j] in _MODEL_SUFFIXES:
                chosen.append(filtered[j])
                j += 1
            return chosen

    distinctive = [t for t in filtered if len(t) >= 4]
    return distinctive[:2] if distinctive else filtered[:2]


def extract_variant_signature(product: Any) -> dict:
    name = normalize(getattr(product, "name", None))
    combined = _combined_text(product)
    category = normalize(getattr(product, "category", None))
    brand = normalize(getattr(product, "brand", None)) or None

    name_text, short_text, combined = _product_text_parts(product)
    capacities = _extract_capacities(combined)
    name_capacities = _extract_capacities(name_text)
    short_capacities = _extract_capacities(short_text)
    specs = _extract_hard_specs(combined)
    model = " ".join(_model_tokens(combined, brand)) or None

    # Exact hardware identity is deliberately kept separate from generic model
    # wording so RTX 4050/4060, CPU models, etc. cannot collapse together.
    gpu_models = sorted({f"rtx{m.group(1)}{('_' + m.group(2).lower()) if m.group(2) else ''}" for m in re.finditer(r"\brtx\s*[- ]?(\d{4})(?:\s*(ti|super))?\b", combined, re.I)})
    cpu_models = sorted({f"i{m.group(1)}-{m.group(2)}" for m in re.finditer(r"\bi([3579])\s*[- ]?(\d{4,5})[a-z]*\b", combined, re.I)})
    ryzen_models = sorted({f"ryzen{m.group(1)}-{m.group(2)}" for m in re.finditer(r"\bryzen\s*([3579])\s*[- ]?(\d{4})[a-z]*\b", combined, re.I)})

    return {
        "color": _extract_color(combined),
        "capacities": capacities,
        "name_capacities": name_capacities,
        "short_description_capacities": short_capacities,
        "capacity": capacities[0] if capacities else None,  # backward compatibility
        "screen": _extract_screen(combined),
        "size": None,
        "model": model,
        "gpu_models": gpu_models,
        "cpu_models": cpu_models,
        "ryzen_models": ryzen_models,
        "hard_specs": specs,
        "category": category,
    }


def _present_values(sig: dict, key: str) -> set[str]:
    value = sig.get(key)
    if isinstance(value, list):
        return {str(x) for x in value if x}
    if value:
        return {str(value)}
    return set()


def _same_nonempty(a: set[str], b: set[str]) -> bool:
    """Compare variant values while allowing one listing to omit a spec."""
    if not a or not b:
        return True
    if a == b:
        return True
    if a < b or b < a:
        return True
    return False


def signatures_match(a: dict, b: dict) -> bool:
    """Return False only for a known variant/identity contradiction.

    Colour and screen size are intentionally non-decisive because the existing
    project semantics treat them as listing modifiers. Capacity, exact hardware
    model, cable length, wattage, RAM speed and furniture count are decisive.
    """
    if not _same_nonempty(_present_values(a, "capacities"), _present_values(b, "capacities")):
        return False

    for key in ("gpu_models", "cpu_models", "ryzen_models"):
        if not _same_nonempty(_present_values(a, key), _present_values(b, key)):
            return False

    hard_a = a.get("hard_specs", {}) or {}
    hard_b = b.get("hard_specs", {}) or {}
    for key in ("length", "wattage", "speed", "rpm", "chair_count"):
        if not _same_nonempty(set(hard_a.get(key, [])), set(hard_b.get(key, []))):
            return False

    model_a = a.get("model")
    model_b = b.get("model")
    if model_a and model_b and normalize(model_a) != normalize(model_b):
        # A digit-bearing model is a strong identity. Non-numeric marketing
        # phrases (e.g. "salle manger") are intentionally left to lexical
        # similarity because stores often phrase them differently.
        if any(ch.isdigit() for ch in normalize(model_a)) or any(ch.isdigit() for ch in normalize(model_b)):
            return False

    return True


def signature_diff(a: dict, b: dict) -> list[str]:
    diff: list[str] = []
    ca, cb = _present_values(a, "capacities"), _present_values(b, "capacities")
    if ca and cb and not _same_nonempty(ca, cb):
        diff.append(f"capacity:{sorted(ca)}!={sorted(cb)}")

    for key, label in (("gpu_models", "gpu"), ("cpu_models", "cpu"), ("ryzen_models", "ryzen")):
        va, vb = _present_values(a, key), _present_values(b, key)
        if va and vb and va != vb:
            diff.append(f"{label}:{sorted(va)}!={sorted(vb)}")

    hard_a, hard_b = a.get("hard_specs", {}) or {}, b.get("hard_specs", {}) or {}
    for key in ("length", "wattage", "speed", "rpm", "chair_count"):
        va, vb = set(hard_a.get(key, [])), set(hard_b.get(key, []))
        if va and vb and va != vb:
            diff.append(f"{key}:{sorted(va)}!={sorted(vb)}")

    ma, mb = normalize(a.get("model")), normalize(b.get("model"))
    if ma and mb and ma != mb and (any(ch.isdigit() for ch in ma) or any(ch.isdigit() for ch in mb)):
        diff.append(f"model:{ma}!={mb}")

    return diff


def build_base_signature(brand: Optional[str], name: Optional[str], category: Optional[str]) -> str:
    base_name = _strip_attributes(normalize(name))
    parts = [normalize(brand), base_name, normalize(category)]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _strip_attributes(text: str) -> str:
    text = normalize(text)
    for alias in sorted(_color_lookup(), key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(alias)}(?!\w)", " ", text)
    text = _CAPACITY_RE.sub(" ", text)
    text = _SCREEN_RE.sub(" ", text)
    text = _WATT_RE.sub(" ", text)
    text = _SPEED_RE.sub(" ", text)
    text = _LENGTH_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
