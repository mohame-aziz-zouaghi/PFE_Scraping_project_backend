"""Deterministic cross-site product matching rules.

The rule order is deliberate:
    1. Same-site guard
    2. Brand compatibility (missing brand is unknown, not a mismatch)
    3. Category compatibility
    4. Hard variant/identity contradictions
    5. Strong exact identity acceptance
    6. Weighted lexical similarity
    7. Borderline -> LLM verification by the caller
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.core.product_tokenizer import load_keywords, normalize, tokenize_product
from app.core.variant_signature import extract_variant_signature, signature_diff, signatures_match


@dataclass
class MatchDecision:
    decision: str
    reason: str
    confidence: float
    token_overlap: float
    brand_match: bool
    category_match: bool
    variant_match: bool
    price_ratio: Optional[float] = None


_BRAND_SYNONYMS: dict[str, set[str]] = {
    "hp": {"hewlett packard", "hp inc", "hp inc."},
    "samsung": {"samsung electronics"},
    "asus": {"asustek", "asustek computer"},
    "lenovo": {"lenovo group"},
    "lg": {"lg electronics"},
    "apple": {"apple inc", "apple computer"},
    "msi": {"micro star international", "micro-star international"},
}

_CATEGORY_BUCKETS: dict[str, tuple[str, ...]] = {
    "laptop": ("laptop", "pc portable", "ordinateur portable", "notebook", "portable"),
    "desktop": ("desktop", "pc bureau", "pc de bureau", "tour", "ordinateur de bureau", "pc gamer", "setup"),
    "smartphone": ("smartphone", "telephone", "phone", "gsm", "mobile"),
    "tablet": ("tablet", "tablette"),
    "monitor": ("monitor", "ecran", "screen", "display"),
    "printer": ("printer", "imprimante", "photocopieur", "multifonction"),
    "cable": ("cable", "adaptateur", "hdmi", "usb"),
    "case": ("case", "coque", "etui", "cover", "housse", "protection"),
    "charger": ("charger", "chargeur", "alimentation", "powerbank", "adaptateur secteur"),
    "mouse": ("souris", "mouse"),
    "keyboard": ("clavier", "keyboard"),
    "headphone": ("casque", "ecouteur", "headphone", "earbud", "buds"),
    "speaker": ("haut-parleur", "haut parleur", "speaker", "enceinte"),
    "storage": ("ssd", "disque dur", "hdd", "cle usb", "carte memoire"),
    "camera": ("camera", "appareil photo", "objectif", "webcam"),
    "tv": ("tv", "television", "televiseur"),
    "cooling": ("watercooling", "aio", "refroidissement", "cooler", "ventilateur", "fan"),
    "gpu": ("carte graphique", "gpu", "geforce", "radeon", "rtx", "gtx"),
    "cpu": ("processeur", "cpu", "ryzen", "intel core"),
    "motherboard": ("carte mere", "motherboard"),
    "psu": ("alimentation pc", "psu", "power supply"),
    "chair": ("chaise", "fauteuil", "siege"),
    "furniture": ("table", "bureau", "meuble", "chaise", "fauteuil"),
    "appliance": ("refrigerateur", "frigo", "lave-linge", "machine a laver", "lave vaisselle", "micro onde", "four", "aspirateur", "mixeur", "blender", "cafetière", "cuisine"),
}


def _category_bucket(cat: Optional[str]) -> Optional[str]:
    c = normalize(cat)
    if not c:
        return None
    for bucket, terms in _CATEGORY_BUCKETS.items():
        if any(term in c for term in terms):
            return bucket
    return None

def _price_ratio(a: Any, b: Any) -> Optional[float]:
    """
    Compute the ratio between the effective prices of two products.
    Returns None if either price is missing.
    """
    pa = getattr(a, "price", None)
    pb = getattr(b, "price", None)
    if pa is None or pb is None:
        return None
    try:
        pa = float(pa)
        pb = float(pb)
    except (TypeError, ValueError):
        return None
    if pa <= 0 or pb <= 0:
        return None
    lo, hi = sorted([pa, pb])
    return hi / lo


def _price_ok(a: Any, b: Any) -> bool:
    """
    True if the two products could plausibly be the same product
    based on price alone. Missing price → don't gate (return True).
    """
    ratio = _price_ratio(a, b)
    if ratio is None:
        return True
    return ratio <= _MAX_PRICE_RATIO


_MAX_PRICE_RATIO = 2.0


def _brands_compatible(a: str, b: str) -> bool:
    if not a or not b or a == b:
        return True
    if a in b or b in a:
        return True
    for canonical, synonyms in _BRAND_SYNONYMS.items():
        if (a == canonical and b in synonyms) or (b == canonical and a in synonyms):
            return True
    return False


def _categories_compatible(a: Optional[str], b: Optional[str]) -> bool:
    ba, bb = _category_bucket(a), _category_bucket(b)
    return ba is None or bb is None or ba == bb


def _weighted_jaccard(
    a: dict[str, float],
    b: dict[str, float],
    *,
    corpus_token_weights: Optional[dict[str, float]] = None,
) -> float:
    if not a and not b:
        return 0.0
    keys = set(a) | set(b)
    if not keys:
        return 0.0

    def weight(token: str, value: float) -> float:
        if not corpus_token_weights or token not in corpus_token_weights:
            return value
        freq = max(0.0, float(corpus_token_weights[token]))
        idf = 1.0 / (1.0 + math.log1p(freq * 1000.0))
        return value * (1.0 + idf)

    inter = sum(weight(t, min(a.get(t, 0.0), b.get(t, 0.0))) for t in keys if t in a and t in b)
    union = sum(weight(t, max(a.get(t, 0.0), b.get(t, 0.0))) for t in keys)
    return inter / union if union else 0.0


def _desc_to_str(desc: Any) -> Optional[str]:
    if desc is None:
        return None
    if isinstance(desc, dict):
        value = " ".join(str(v) for v in desc.values() if v)
    elif isinstance(desc, list):
        value = " ".join(str(v) for v in desc if v)
    else:
        value = str(desc)
    value = normalize(value)
    return value or None


def _description_overlap(a: str, b: str) -> float:
    kw = load_keywords()
    stop = {normalize(x) for x in kw.get("stopwords", [])}
    spec = {normalize(x) for x in kw.get("spec_keywords", [])}

    def tokens(value: str) -> set[str]:
        raw = re.findall(r"[a-z0-9]+", normalize(value))
        return {t for t in raw if t not in stop and (len(t) > 2 or t in spec)}

    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def _exact_model_match(sig_a: dict, sig_b: dict) -> bool:
    ma, mb = normalize(sig_a.get("model")), normalize(sig_b.get("model"))
    if not ma or not mb or ma != mb:
        return False
    # Exact model alone is enough only when there is no hard contradiction.
    return signatures_match(sig_a, sig_b)


def compare_products(*, a: Any, b: Any, corpus_token_weights: Optional[dict[str, float]] = None) -> MatchDecision:
    site_a = normalize(getattr(a, "site", None))
    site_b = normalize(getattr(b, "site", None))
    if site_a and site_b and site_a == site_b:
        return _reject("same site: cross-site matching requires different sites", brand_match=True, category_match=True, variant_match=False)

    brand_a = normalize(getattr(a, "brand", None))
    brand_b = normalize(getattr(b, "brand", None))
    if not _brands_compatible(brand_a, brand_b):
        return _reject(f"brand mismatch: {brand_a!r} vs {brand_b!r}", brand_match=False)

    category_a = getattr(a, "category", None)
    category_b = getattr(b, "category", None)
    if not _categories_compatible(category_a, category_b):
        return _reject(f"category mismatch: {category_a!r} vs {category_b!r}", category_match=False)

    if not _price_ok(a, b):
        ratio = _price_ratio(a, b)
        return _reject(
            f"price mismatch: ratio {ratio:.2f} exceeds "
            f"{_MAX_PRICE_RATIO}×",
            brand_match=True,
            category_match=True,
            variant_match=True,
        )

    sig_a = extract_variant_signature(a)
    sig_b = extract_variant_signature(b)
    if not signatures_match(sig_a, sig_b):
        diff = signature_diff(sig_a, sig_b)
        return _reject(
            "decisive variant mismatch: " + (", ".join(diff) if diff else "hard specification conflict"),
            variant_match=False,
        )

    ts_a = tokenize_product(name=getattr(a, "name", None), brand=getattr(a, "brand", None), category=category_a)
    ts_b = tokenize_product(name=getattr(b, "name", None), brand=getattr(b, "brand", None), category=category_b)
    overlap = _weighted_jaccard(ts_a.model_tokens, ts_b.model_tokens, corpus_token_weights=corpus_token_weights)

    # Strong identity: same normalized model and compatible hard specs.
    if _exact_model_match(sig_a, sig_b):
        # Exact normalized model identity is stronger than raw lexical overlap.
        # Listings can contain extra CPU/storage marketing details on only one
        # side, so expose an identity-aware overlap floor for callers/tests.
        identity_overlap = max(overlap, 0.75)
        return MatchDecision(
            "accept",
            f"exact model identity: {sig_a['model']}",
            max(0.95, identity_overlap),
            identity_overlap,
            True,
            True,
            True,
        )

    # Exact hardware identifiers can rescue listings whose marketing model text differs.
    hardware_keys = ("gpu_models", "cpu_models", "ryzen_models")
    hardware_exact = False
    for key in hardware_keys:
        va = set(sig_a.get(key, []))
        vb = set(sig_b.get(key, []))
        if va and vb:
            hardware_exact = va == vb
            if not hardware_exact:
                return _reject(f"hardware identity mismatch: {sorted(va)} vs {sorted(vb)}", variant_match=False)

    desc_a = _desc_to_str(getattr(a, "short_description", None))
    desc_b = _desc_to_str(getattr(b, "short_description", None))
    desc_overlap = _description_overlap(desc_a, desc_b) if desc_a and desc_b else None
    confidence = overlap if desc_overlap is None else (0.85 * overlap + 0.15 * desc_overlap)

    # Exact non-identity specifications (for example a 100W USB-C cable)
    # can be enough when the lexical representation is clearly the same.
    hard_a = sig_a.get("hard_specs", {}) or {}
    hard_b = sig_b.get("hard_specs", {}) or {}
    hard_keys = ("length", "wattage", "speed", "rpm", "chair_count")
    hard_pairs = [(set(hard_a.get(k, [])), set(hard_b.get(k, []))) for k in hard_keys]
    hard_present = [(x, y) for x, y in hard_pairs if x or y]
    hard_exact = bool(hard_present) and all(x and y and x == y for x, y in hard_present)
    if hard_exact and confidence >= 0.30:
        return MatchDecision("accept", f"matching hard specifications; overlap={overlap:.2f}", max(confidence, 0.80), overlap, True, True, True)

    # Hardware exactness + reasonable lexical agreement is strong evidence.
    if hardware_exact and confidence >= 0.45:
        return MatchDecision("accept", f"exact hardware identity; overlap={overlap:.2f}", max(confidence, 0.85), overlap, True, True, True)

    kw = load_keywords()
    accept_thr = float(kw.get("accept_threshold", 0.70))
    borderline_thr = float(kw.get("borderline_threshold", 0.40))
    ratio = _price_ratio(a, b)
    if confidence >= accept_thr:
        return MatchDecision("accept", f"token overlap={overlap:.2f}", confidence, overlap, True, True, True)
    if confidence >= borderline_thr:
        return MatchDecision("borderline", f"weak overlap={overlap:.2f}", confidence, overlap, True, True, True)
    return MatchDecision("reject", f"insufficient overlap={overlap:.2f}", confidence, overlap, True, True, True)


def _reject(reason: str, *, brand_match: bool = True, category_match: bool = True, variant_match: bool = True) -> MatchDecision:
    return MatchDecision("reject", reason, 0.0, 0.0, brand_match, category_match, variant_match)
