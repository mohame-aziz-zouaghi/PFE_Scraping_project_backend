# app/services/rag_intents.py
"""
Intent shortcuts — hardcoded answers for the top 10 questions.

These skip retrieval and generation entirely. When a user's
message matches one of these patterns, the answer is returned
in <1ms. Zero LLM cost, zero hallucination risk.

This is the "80% of volume from 20% of questions" layer. Every
pattern here should be based on real traffic from
user_events — add new ones as you see new frequent intents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Intent:
    name: str
    patterns: tuple[re.Pattern, ...]
    answer: str
    category: str    # for the "browse more" link


# Pre-compile patterns for speed (regex is run on every turn
# before retrieval, so it must be cheap).
def _p(*patterns: str) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


INTENTS: tuple[Intent, ...] = (
    Intent(
        name="sign_in",
        patterns=_p(
            r"\b(sign\s*in|signin|log\s*in|login)\b",
            r"\bhow\s+(do\s+i|to|can\s+i)\s+(sign|log)\b",
        ),
        answer=(
            "To sign in, click 'Sign in' in the top-right corner "
            "of any page. If you don't have an account yet, choose "
            "'Create an account' on the same screen — it's free and "
            "takes about 30 seconds. Signing in unlocks unlimited "
            "chat with the assistant, saved wishlists, and price "
            "alerts across all your devices."
        ),
        category="account",
    ),
    Intent(
        name="create_account",
        patterns=_p(
            r"\b(create|make|open|start)\s+(an\s+)?account\b",
            r"\b(sign\s*up|signup|register|registration)\b",
        ),
        answer=(
            "Creating an account is free. Click 'Sign in' in the "
            "top-right corner, then choose 'Create an account'. "
            "You'll need an email address and a password. Once "
            "signed up, you can save wishlists, set price alerts, "
            "and chat with the assistant without the 7-message "
            "anonymous limit."
        ),
        category="account",
    ),
    Intent(
        name="free_cost",
        patterns=_p(
            r"\b(is|isn'?t|are)\s+(it|this|the\s+site|the\s+service)\s+free\b",
            r"\bhow\s+much\s+does\s+(it|this|the\s+site)\s+cost\b",
            r"\b(do|must|need\s+to)\s+i\s+pay\b",
            r"\bcost\s+(of|for)\s+(the\s+)?service\b",
        ),
        answer=(
            "Yes, the site is completely free to use. Searching, "
            "comparing prices, and browsing the catalog cost "
            "nothing. Creating an account is also free. Our only "
            "revenue comes from affiliate fees when you buy a "
            "product through our link — the price you pay is the "
            "same as going to the store directly."
        ),
        category="other",
    ),
    Intent(
        name="contact_support",
        patterns=_p(
            r"\b(contact|support|reach|email)\s*(us|you|team|site)?\b",
            r"\bcustomer\s+service\b",
            r"\bhow\s+(do\s+i|to|can\s+i)\s+contact\b",
        ),
        answer=(
            "You can reach us at contact@this-site.tn for any "
            "issue — wrong prices, missing products, or bugs. For "
            "problems with an order placed on a specific store, "
            "please contact that store directly; we don't process "
            "purchases and can't look up your order."
        ),
        category="other",
    ),
    Intent(
        name="sites_covered",
        patterns=_p(
            r"\b(which|what)\s+(sites|stores|shops|retailers|websites)\b",
            r"\bwhere\s+do\s+you\s+compare\b",
        ),
        answer=(
            "We aggregate products from six Tunisian retailers: "
            "SBS, Mytek, Tunisianet, MegaPC, Spacenet, and Skymil. "
            "For every product, you see all listings across these "
            "stores with current prices and stock. If a store "
            "doesn't appear for a product, it currently doesn't "
            "carry it."
        ),
        category="general",
    ),
    Intent(
        name="wishlist",
        patterns=_p(
            r"\b(wishlist|favorites|favourites|saved\s+items)\b",
            r"\bhow\s+do\s+i\s+(save|add\s+to\s+wishlist)\b",
        ),
        answer=(
            "To add a product to your wishlist, open any product "
            "page and click the heart icon next to the product "
            "name. Your wishlist is saved to your account, so it's "
            "available on any device once you sign in. You can "
            "view it under 'My account' → 'Wishlist', and export "
            "it as CSV or JSON."
        ),
        category="account",
    ),
    Intent(
        name="price_alert",
        patterns=_p(
            r"\b(price\s+alert|alert\s+me|notify\s+me|notification)\b",
            r"\bhow\s+do\s+i\s+(get|set)\s+(an?\s+)?alert\b",
        ),
        answer=(
            "On any product page, click 'Set alert' and choose "
            "your target price. We check the price every few "
            "minutes and notify you by email or in-app the moment "
            "it drops to or below your target. You can have up to "
            "50 active alerts per account. Manage them under "
            "'My account' → 'Alerts'."
        ),
        category="pricing",
    ),
    Intent(
        name="availability",
        patterns=_p(
            r"\b(in\s+stock|out\s+of\s+stock|available|availability)\b",
            r"\b(on\s+order|backorder|pre.?order)\b",
        ),
        answer=(
            "'In stock' means the store has the product physically "
            "available right now. 'On order' means it's confirmed "
            "with the supplier but not yet on the shelf — typically "
            "5 to 15 business days. 'Out of stock' means it can't "
            "be ordered at the moment. Every product page shows "
            "the current status for every store."
        ),
        category="availability",
    ),
    Intent(
        name="report_wrong_price",
        patterns=_p(
            r"\b(wrong\s+price|incorrect\s+price|price\s+is\s+wrong)\b",
            r"\breport\s+(a\s+)?(wrong|bad|incorrect|error)\b",
        ),
        answer=(
            "If a price looks wrong, click the small flag icon "
            "next to it on the product page and describe the "
            "issue. We re-check against the source store within "
            "15 minutes. If the price was wrong, it gets corrected "
            "automatically. You can track the status of any report "
            "under 'My account' → 'Reports'."
        ),
        category="other",
    ),
    Intent(
        name="recommendations",
        patterns=_p(
            r"\b(recommend|recommendation|recommended|suggest)\b",
            r"\bhow\s+(are|do)\s+(recommendations?|you\s+recommend)\b",
        ),
        answer=(
            "Recommendations combine three signals: your recent "
            "activity (searches, wishlists, views from the last "
            "14 days, with fresher activity weighted more), your "
            "long-term profile (categories and brands you browse "
            "regularly), and what's popular and in stock. Fresh "
            "wishlists and searches appear at the top of your "
            "feed automatically. You can reset your profile under "
            "'My account' → 'Preferences'."
        ),
        category="recommendation",
    ),
)


# Pre-compile the "on-domain" pattern used by fallback generation
# (Strategy 3). If a message matches any of these, we let the LLM
# try to answer from general shopping knowledge even when
# retrieval misses.
_ON_DOMAIN_PATTERNS: tuple[re.Pattern, ...] = _p(
    r"\b(product|article|item|model|reference|sku)\b",
    r"\b(price|cost|cheap|expensive|discount|deal|sale|promo)\b",
    r"\b(buy|purchase|order|cart|checkout)\b",
    r"\b(ship|deliver|stock|available|supplier)\b",
    r"\b(brand|manufacturer|warranty|guarantee)\b",
    r"\b(laptop|phone|tablet|monitor|headphone|keyboard|mouse|"
    r"tv|console|console|camera|charger|cable|speaker)\b",
    r"\b(account|login|sign\s*in|register|profile|password)\b",
    r"\b(wishlist|favorite|favourite|alert|notification)\b",
    r"\b(compare|comparison|difference|versus|vs)\b",
)


def match_intent(message: str) -> Optional[Intent]:
    """Return the first matching intent, or None."""
    for intent in INTENTS:
        for pattern in intent.patterns:
            if pattern.search(message):
                return intent
    return None


def is_on_domain(message: str) -> bool:
    """
    True if the message looks like a shopping-site question.

    Requires at least 2 distinct on-domain pattern hits — a
    single match is too easy to trigger accidentally
    ("whats the weather today" would match "today" if we had
    a "today" pattern, which we don't, but this defensive
    threshold is here in case future patterns get broader).
    """
    hits = sum(
        1 for pattern in _ON_DOMAIN_PATTERNS
        if pattern.search(message)
    )
    return hits >= 2


_OFF_DOMAIN_PATTERNS: tuple[re.Pattern, ...] = _p(
    r"\b(weather|temperature|forecast|rain|sunny|cloudy)\b",
    r"\b(joke|funny|meme|story|poem|riddle)\b",
    r"\b(news|politics|sport|football|match)\b",
    r"\b(code|python|javascript|coding|programming)\b",
    r"\b(recipe|food|cook|restaurant)\b",
    r"\b(history|war|religion|politics)\b",
    r"\b(time|date|day)\s+(now|today|is it)\b",
    r"\bwho\s+(is|are|was)\b",
    r"\btranslate\b",
    r"\bdefine\b",
)


def is_off_domain(message: str) -> bool:
    """
    True if the message is clearly NOT about shopping.
    Used as a hard gate before fallback generation — even if
    is_on_domain() accidentally matches something, an explicit
    off-domain signal wins.
    """
    for pattern in _OFF_DOMAIN_PATTERNS:
        if pattern.search(message):
            return True
    return False
