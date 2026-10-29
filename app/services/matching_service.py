"""Cross-site product matching service.

Two entry points:

  run_full_match        — the initial/periodic full matching pass.
                          Candidate pairs via pgvector, deterministic
                          rules, LLM for borderline, Union-Find
                          clustering, base grouping, price rollup.

  run_targeted_match    — the verification loop's rebuild path.
                          Delegates to CanonicalRebuilder, which uses
                          the SAME deterministic identity_signature_hash
                          logic as scripts/backfill_canonical_products.py.
                          No pairwise matching, no LLM — those decisions
                          were already made during initial ingestion.
"""
from __future__ import annotations

import asyncio
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.llm_client import LLMClient
from app.core.logging_config import get_logger
from app.core.matching_config import (
    BASE,
    MATCH_VERIFIER_CACHE_NAMESPACE,
    MATCH_VERIFIER_PROMPT_VERSION,
    VARIANT,
)
from app.core.variant_signature import (
    _strip_attributes,
    build_base_signature,
    extract_variant_signature,
    signature_diff,
    signatures_match,
)
from app.models.product import Product
from app.repositories.canonical_repo import CanonicalRepo
from app.repositories.match_repo import MatchRepo
from app.services.matching_helpers import (
    recompute_base_cheapest,
    recompute_canonical_cheapest,
)

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Verdict schema
# ═══════════════════════════════════════════════════════
class MatchVerdict(BaseModel):
    same_product: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=400)


# ═══════════════════════════════════════════════════════
# Union-Find
# ═══════════════════════════════════════════════════════
class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[UUID, UUID] = {}
        self._rank: dict[UUID, int] = {}

    def find(self, x: UUID) -> UUID:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            return x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: UUID, b: UUID) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def members(self, root: UUID) -> list[UUID]:
        return [n for n in self._parent if self.find(n) == root]

    def clusters(self) -> dict[UUID, list[UUID]]:
        out: dict[UUID, list[UUID]] = defaultdict(list)
        for node in self._parent:
            out[self.find(node)].append(node)
        for v in out.values():
            v.sort(key=str)
        return dict(out)


# ═══════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════
@dataclass
class MatchingStats:
    candidates_found: int = 0
    auto_accepted: int = 0
    rule_rejected: int = 0
    variant_rejected: int = 0
    llm_checked: int = 0
    llm_accepted: int = 0
    llm_rejected: int = 0
    llm_errors: int = 0
    llm_deferred: int = 0
    cluster_rejected: int = 0
    variant_groups_created: int = 0
    products_grouped: int = 0
    base_groups_created: int = 0
    base_groups_updated: int = 0
    pairs_linked: int = 0
    integrity_errors: int = 0
    errors: int = 0
    details: dict = field(default_factory=dict)

    _FIXED_KEYS = {
        "candidates_found", "auto_accepted", "rule_rejected",
        "variant_rejected", "llm_checked", "llm_accepted",
        "llm_rejected", "llm_errors", "llm_deferred",
        "cluster_rejected", "variant_groups_created",
        "products_grouped", "base_groups_created",
        "base_groups_updated", "pairs_linked", "integrity_errors",
        "errors",
    }

    def as_dict(self) -> dict:
        out = {k: getattr(self, k) for k in self._FIXED_KEYS}
        for k, v in self.details.items():
            if k in out:
                raise ValueError(f"details key collides with counter: {k}")
            out[k] = v
        return out


# ═══════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════
class MatchingService:
    def __init__(
        self, db: AsyncSession, *, model: Optional[str] = None,
    ) -> None:
        self.db = db
        self.canon_repo = CanonicalRepo(db)
        self.match_repo = MatchRepo(db)
        self.model = model or settings.active_embedding_model
        self._llm_sem = asyncio.Semaphore(VARIANT.LLM_MAX_CONCURRENCY)

    # ═══════════════════════════════════════════════════════
    # Entry point — full match
    # ═══════════════════════════════════════════════════════
    async def run_full_match(
        self,
        *,
        max_distance: float = VARIANT.LLM_VERIFY_MAX_DISTANCE,
        max_candidates: Optional[int] = None,
        llm_budget: Optional[int] = None,
        clear_existing: bool = True,
    ) -> MatchingStats:
        stats = MatchingStats()
        logger.info(
            "matching_started", model=self.model, llm_budget=llm_budget,
        )

        if clear_existing:
            await self.canon_repo.clear_all()

        candidates = await self.match_repo.find_candidate_pairs(
            model=self.model, max_distance=max_distance,
        )
        if max_candidates is not None:
            candidates = candidates[: max(0, max_candidates)]
        stats.candidates_found = len(candidates)

        if not candidates:
            await recompute_canonical_cheapest(self.db)
            await recompute_base_cheapest(self.db)
            await self.canon_repo.sync_embedding_canonical_ids()
            await self.db.commit()
            return stats

        return await self._execute_pipeline(
            candidates=candidates,
            llm_budget=llm_budget,
            stats=stats,
        )

    # ═══════════════════════════════════════════════════════
    # Entry point — targeted match (deterministic rebuild)
    # ═══════════════════════════════════════════════════════
    async def run_targeted_match(
        self,
        *,
        product_ids: list[UUID],
        max_distance: float = VARIANT.LLM_VERIFY_MAX_DISTANCE,
        llm_budget: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> MatchingStats:
        """
        Rebuild canonicals + bases for a subset of products using
        the SAME deterministic identity_signature_hash logic as the
        initial backfill. No pairwise matching, no LLM.

        max_distance and llm_budget are kept in the signature for
        backwards compatibility but are unused.
        """
        from app.services.canonical_rebuilder import (
            CanonicalRebuilder,
        )

        stats = MatchingStats()
        logger.info(
            "targeted_rebuild_started",
            product_count=len(product_ids),
            mode="deterministic_rebuild",
            task_id=task_id,
        )

        if not product_ids:
            return stats

        await self._ensure_embeddings(product_ids)

        rebuilder = CanonicalRebuilder(self.db)
        rebuild_stats = await rebuilder.rebuild_for_products(
            product_ids,
            task_id=task_id,
        )

        stats.candidates_found = rebuild_stats["products"]
        stats.variant_groups_created = (
            rebuild_stats["canonicals_created"]
        )
        stats.products_grouped = (
            rebuild_stats["canonicals_created"]
            + rebuild_stats["canonicals_updated"]
        )
        stats.base_groups_created = rebuild_stats["bases_created"]
        stats.base_groups_updated = rebuild_stats["bases_updated"]
        stats.errors = 0

        logger.info(
            "targeted_rebuild_complete",
            **rebuild_stats,
        )
        return stats

    # ═══════════════════════════════════════════════════════
    # Shared pipeline (used by run_full_match only)
    # ═══════════════════════════════════════════════════════
    async def _execute_pipeline(
        self,
        *,
        candidates: list[dict],
        llm_budget: Optional[int],
        stats: MatchingStats,
        targeted: bool = False,
    ) -> MatchingStats:
        # ── 1. Corpus weights
        from app.services.corpus_stats import compute_token_frequencies
        corpus_weights = await compute_token_frequencies(self.db)

        # ── 2. Load products + signatures + embeddings
        all_pids: set[UUID] = set()
        for c in candidates:
            all_pids.add(c["product_a_id"])
            all_pids.add(c["product_b_id"])

        rows = (await self.db.execute(
            select(Product).where(Product.id.in_(all_pids))
        )).scalars().all()
        product_map: dict[UUID, Product] = {p.id: p for p in rows}
        signatures = {
            pid: extract_variant_signature(p)
            for pid, p in product_map.items()
        }
        await self._load_embeddings(list(all_pids))

        from app.services.matching_rules import compare_products

        # ── 3. Rule + LLM adjudication
        accepted_edges: list[tuple[UUID, UUID, float]] = []
        llm_calls_made = 0

        for c in candidates:
            pid_a, pid_b = c["product_a_id"], c["product_b_id"]
            dist = float(c["cosine_distance"])
            pa, pb = product_map.get(pid_a), product_map.get(pid_b)
            if pa is None or pb is None:
                stats.errors += 1
                continue

            decision = compare_products(
                a=pa, b=pb, corpus_token_weights=corpus_weights,
            )

            if decision.decision == "reject":
                if not decision.variant_match:
                    stats.variant_rejected += 1
                else:
                    stats.rule_rejected += 1
                await self.match_repo.upsert_pair(
                    product_a_id=pid_a, product_b_id=pid_b, group_id=None,
                    cosine_distance=dist, decision="rejected",
                    decision_source="rule",
                    variant_match=decision.variant_match,
                    token_overlap=decision.token_overlap,
                    llm_reason=decision.reason[:400],
                )
                continue

            if decision.decision == "accept":
                stats.auto_accepted += 1
                accepted_edges.append((pid_a, pid_b, decision.confidence))
                await self.match_repo.upsert_pair(
                    product_a_id=pid_a, product_b_id=pid_b, group_id=None,
                    cosine_distance=dist, decision="accepted",
                    decision_source="rule",
                    variant_match=True,
                    token_overlap=decision.token_overlap,
                    llm_reason=decision.reason[:400],
                    llm_confidence=decision.confidence,
                )
                continue

            # Borderline -> LLM
            if not settings.matching_llm_enabled:
                stats.llm_deferred += 1
                await self.match_repo.upsert_pair(
                    product_a_id=pid_a, product_b_id=pid_b, group_id=None,
                    cosine_distance=dist, decision="rejected",
                    decision_source="config",
                    variant_match=True,
                    token_overlap=decision.token_overlap,
                    llm_reason="LLM verification disabled by config.",
                )
                continue

            if llm_budget is not None and llm_calls_made >= llm_budget:
                stats.llm_deferred += 1
                await self.match_repo.upsert_pair(
                    product_a_id=pid_a, product_b_id=pid_b, group_id=None,
                    cosine_distance=dist,
                    decision="borderline_deferred",
                    decision_source="budget",
                    variant_match=True,
                    token_overlap=decision.token_overlap,
                    llm_reason="LLM budget exhausted; deferred.",
                )
                continue

            async with self._llm_sem:
                verdict = await self._llm_verify(
                    pa, pb, signatures[pid_a], signatures[pid_b], dist,
                )
            llm_calls_made += 1
            stats.llm_checked += 1

            if verdict is None:
                stats.llm_errors += 1
                stats.errors += 1
                await self.match_repo.upsert_pair(
                    product_a_id=pid_a, product_b_id=pid_b, group_id=None,
                    cosine_distance=dist, decision="error",
                    decision_source="llm",
                    variant_match=True,
                    token_overlap=decision.token_overlap,
                    llm_reason="LLM verification failed; pair not accepted.",
                )
                continue

            accepted = (
                verdict.same_product
                and verdict.confidence >= VARIANT.MIN_EDGE_CONFIDENCE
            )
            if accepted:
                stats.llm_accepted += 1
                accepted_edges.append((pid_a, pid_b, verdict.confidence))
                await self.match_repo.upsert_pair(
                    product_a_id=pid_a, product_b_id=pid_b, group_id=None,
                    cosine_distance=dist, decision="accepted",
                    decision_source="llm", variant_match=True,
                    token_overlap=decision.token_overlap,
                    llm_reason=verdict.reason,
                    llm_confidence=verdict.confidence,
                )
            else:
                stats.llm_rejected += 1
                await self.match_repo.upsert_pair(
                    product_a_id=pid_a, product_b_id=pid_b, group_id=None,
                    cosine_distance=dist, decision="rejected",
                    decision_source="llm", variant_match=True,
                    token_overlap=decision.token_overlap,
                    llm_reason=verdict.reason,
                    llm_confidence=verdict.confidence,
                )

        # ── 4. Constrained clustering
        uf = UnionFind()
        for a, b, _ in sorted(
            accepted_edges, key=lambda x: (-x[2], str(x[0]), str(x[1])),
        ):
            if self._can_merge_variant_components(
                uf, a, b, product_map, signatures,
            ):
                uf.union(a, b)
            else:
                stats.cluster_rejected += 1
                pair = await self.match_repo.get_pair(a, b)
                if pair:
                    await self.match_repo.upsert_pair(
                        product_a_id=a, product_b_id=b, group_id=None,
                        cosine_distance=float(pair.cosine_distance or 0.0),
                        decision="rejected",
                        decision_source="cluster_guard",
                        variant_match=False,
                        token_overlap=float(pair.token_overlap or 0.0),
                        llm_reason=(
                            "Accepted pair conflicts with an existing "
                            "exact-variant component."
                        ),
                    )

        # ── 5. Persist variant clusters
        clusters = uf.clusters()
        product_to_variant_group: dict[UUID, UUID] = {}
        variant_group_ids: list[UUID] = []

        for _root, members in clusters.items():
            if len(members) < 2:
                continue
            products = [product_map[m] for m in members if m in product_map]
            sites = {normalize_site(p.site) for p in products if p.site}
            if len(sites) < 2:
                continue
            if not self._group_is_valid(products, signatures):
                stats.cluster_rejected += 1
                continue

            canonical_name = _most_common(
                [p.name for p in products if p.name], default="Unknown",
            )
            canonical_brand = _most_common(
                [p.brand for p in products if p.brand], default=None,
            )
            canonical_category = _most_common(
                [p.category for p in products if p.category], default=None,
            )
            variant_sig = self._canonical_variant_signature(
                [signatures[m] for m in members]
            )

            edge_confs = [
                conf for a, b, conf in accepted_edges
                if uf.find(a) == uf.find(b) and a in members and b in members
            ]
            avg_conf = (
                sum(edge_confs) / len(edge_confs) if edge_confs else 0.0
            )

            group_id = await self.canon_repo.create_variant_group(
                canonical_name=canonical_name,
                canonical_brand=canonical_brand,
                canonical_category=canonical_category,
                identity_signature=variant_sig,
                variants=None,
                listing_count=len(members),
                site_count=len(sites),
                confidence=round(avg_conf, 4),
            )
            await self.canon_repo.assign_products_to_group(
                members, group_id,
            )
            variant_group_ids.append(group_id)
            for m in members:
                product_to_variant_group[m] = group_id
            stats.variant_groups_created += 1
            stats.products_grouped += len(members)

        # ── 6. Link accepted pairs to their canonical
        for a, b, _ in accepted_edges:
            gid = product_to_variant_group.get(a)
            if gid is None or product_to_variant_group.get(b) != gid:
                continue
            pair = await self.match_repo.get_pair(a, b)
            if pair and pair.canonical_id != gid:
                await self.match_repo.link_pair_to_group(pair.id, gid)
                stats.pairs_linked += 1

        # ── 7. Base grouping
        await self._build_base_groups_deterministic(
            variant_group_ids=variant_group_ids,
            stats=stats,
        )

        # ── 8. Price rollup
        if targeted:
            await recompute_canonical_cheapest(
                self.db, canonical_ids=variant_group_ids,
            )
            await recompute_base_cheapest(self.db)
            await self._recompute_counters_for_canonicals(
                variant_group_ids,
            )
        else:
            await recompute_canonical_cheapest(self.db)
            await recompute_base_cheapest(self.db)

        await self.canon_repo.sync_embedding_canonical_ids()
        await self.db.commit()

        # ── 9. Integrity check
        stats.integrity_errors = await self._validate_integrity()
        stats.errors += stats.integrity_errors

        logger.info(
            "matching_complete",
            targeted=targeted,
            **stats.as_dict(),
        )
        return stats

    # ═══════════════════════════════════════════════════════
    # Clustering guards
    # ═══════════════════════════════════════════════════════
    def _can_merge_variant_components(
        self,
        uf: UnionFind,
        a: UUID,
        b: UUID,
        product_map: dict[UUID, Product],
        signatures: dict[UUID, dict],
    ) -> bool:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return True
        left = uf.members(ra)
        right = uf.members(rb)
        for x in left:
            for y in right:
                px, py = product_map.get(x), product_map.get(y)
                if px is None or py is None:
                    return False
                sx, sy = normalize_site(px.site), normalize_site(py.site)
                if sx and sy and sx == sy:
                    return False
                if not signatures_match(signatures[x], signatures[y]):
                    return False
        return True

    @staticmethod
    def _group_is_valid(
        products: list[Product], signatures: dict[UUID, dict],
    ) -> bool:
        seen_sites: set[str] = set()
        for p in products:
            site = normalize_site(p.site)
            if site and site in seen_sites:
                return False
            if site:
                seen_sites.add(site)
        for i, p in enumerate(products):
            for q in products[i + 1:]:
                if not signatures_match(signatures[p.id], signatures[q.id]):
                    return False
        return len(seen_sites) >= 2

    @staticmethod
    def _canonical_variant_signature(signatures: list[dict]) -> dict:
        if not signatures:
            return {}
        keys = set().union(*(s.keys() for s in signatures))
        result: dict = {}
        for key in keys:
            if key == "color":
                continue
            values = [s.get(key) for s in signatures]
            nonempty = [v for v in values if v not in (None, [], {})]
            if not nonempty:
                continue
            result[key] = _most_common_serializable(nonempty)
        return result

    # ═══════════════════════════════════════════════════════
    # Base grouping
    # ═══════════════════════════════════════════════════════
    async def _build_base_groups_deterministic(
        self,
        *,
        variant_group_ids: list[UUID],
        stats: MatchingStats,
    ) -> None:
        if not variant_group_ids:
            return

        from app.models.canonical_product import CanonicalProduct

        variants = (
            await self.db.execute(
                select(CanonicalProduct).where(
                    CanonicalProduct.id.in_(variant_group_ids)
                )
            )
        ).scalars().all()

        bases: dict[str, list] = defaultdict(list)
        for v in variants:
            sig = build_base_signature(
                v.canonical_brand, v.canonical_name, v.canonical_category,
            )
            bases[sig].append(v)

        for base_sig, children in bases.items():
            existing = await self.canon_repo.get_base_by_signature(base_sig)
            representative = _strip_attributes(
                normalize_text(
                    min((c.canonical_name for c in children), key=len)
                )
            ).strip(" /-|,") or "Unknown"

            variant_ids = [c.id for c in children]
            child_listings = sum(
                (c.listing_count or 0) for c in children
            )

            site_rows = await self.db.execute(
                text(
                    "SELECT DISTINCT site FROM products "
                    "WHERE canonical_id = ANY(:ids) AND site IS NOT NULL"
                ),
                {"ids": variant_ids},
            )
            site_count = sum(1 for _ in site_rows)

            base_id = await self.canon_repo.upsert_base_group(
                id=existing.id if existing else uuid4(),
                base_name=representative,
                base_brand=_most_common(
                    [c.canonical_brand for c in children
                     if c.canonical_brand],
                    default=None,
                ),
                base_category=_most_common(
                    [c.canonical_category for c in children
                     if c.canonical_category],
                    default=None,
                ),
                base_signature=base_sig,
                variant_count=len(children),
                site_count=site_count,
                listing_count=child_listings,
            )
            await self.canon_repo.assign_variants_to_base(
                variant_ids, base_id,
            )
            if existing:
                stats.base_groups_updated += 1
            else:
                stats.base_groups_created += 1

    # ═══════════════════════════════════════════════════════
    # Embeddings
    # ═══════════════════════════════════════════════════════
    async def _ensure_embeddings(
        self, product_ids: list[UUID],
    ) -> int:
        if not product_ids:
            return 0

        existing = set((await self.db.execute(
            text(
                "SELECT product_id FROM product_embeddings "
                "WHERE product_id = ANY(CAST(:ids AS uuid[])) "
                "AND model = :model"
            ),
            {
                "ids": [str(x) for x in product_ids],
                "model": self.model,
            },
        )).scalars().all())

        missing = [pid for pid in product_ids if pid not in existing]
        if not missing:
            return 0

        logger.info(
            "targeted_match_embedding_backfill",
            missing=len(missing),
            model=self.model,
        )

        from app.services.embedding_service import EmbeddingService
        service = EmbeddingService(db=self.db)
        written = await service.embed_products_bulk(missing)
        await self.db.commit()
        return written

    async def _load_embeddings(
        self, product_ids: list[UUID],
    ) -> dict[UUID, list[float]]:
        if not product_ids:
            return {}
        from app.models.product_embedding import ProductEmbedding

        stmt = select(
            ProductEmbedding.product_id, ProductEmbedding.embedding,
        ).where(
            ProductEmbedding.product_id.in_(product_ids),
            ProductEmbedding.model == self.model,
        )
        rows = (await self.db.execute(stmt)).all()
        out: dict[UUID, list[float]] = {}
        dim: Optional[int] = None
        for row in rows:
            vec = list(row.embedding) if row.embedding is not None else []
            if not vec:
                continue
            if dim is None:
                dim = len(vec)
            if len(vec) != dim:
                logger.warning(
                    "embedding_dimension_mismatch",
                    product_id=str(row.product_id),
                    expected=dim, actual=len(vec),
                )
                continue
            out[row.product_id] = vec
        return out

    # ═══════════════════════════════════════════════════════
    # LLM verification (for full match only)
    # ═══════════════════════════════════════════════════════
    async def _llm_verify(
        self,
        a: Product,
        b: Product,
        sig_a: dict,
        sig_b: dict,
        distance: float,
    ) -> Optional[MatchVerdict]:
        payload = {
            "product_a": {
                "site": a.site, "name": a.name, "brand": a.brand,
                "category": a.category,
                "price": float(a.price) if a.price is not None else None,
                "extracted_attributes": sig_a,
            },
            "product_b": {
                "site": b.site, "name": b.name, "brand": b.brand,
                "category": b.category,
                "price": float(b.price) if b.price is not None else None,
                "extracted_attributes": sig_b,
            },
            "embedding_distance": round(distance, 4),
        }
        system = (
            "You are a strict product-matching judge for an e-commerce "
            "price comparison system. Decide whether two listings "
            "represent the same physical product/variant. "
            "Do not infer equivalence when a model, storage, RAM, GPU, "
            "cable length, wattage, speed, colour, or other explicit "
            "specification conflicts. Marketing wording alone does not "
            "make two otherwise identical products different. "
            "When evidence is insufficient, return false. "
            "Reply with JSON only: "
            '{"same_product": bool, "confidence": 0-1, "reason": "..."}'
        )

        pair_key = [
            MATCH_VERIFIER_CACHE_NAMESPACE,
            str(min(a.id, b.id)),
            str(max(a.id, b.id)),
        ]
        client = LLMClient(
            db=self.db,
            endpoint="match_verifier",
            prompt_version=MATCH_VERIFIER_PROMPT_VERSION,
        )

        last_exc: Optional[Exception] = None
        for attempt in range(VARIANT.LLM_RETRY_ATTEMPTS):
            try:
                return await client.chat_json(
                    system=system,
                    user=payload,
                    schema=MatchVerdict,
                    cache_key_parts=pair_key,
                    cache_ttl_seconds=86400 * 7,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "llm_verify_attempt_failed",
                    attempt=attempt + 1,
                    product_a=str(a.id), product_b=str(b.id),
                    error=str(exc)[:200],
                )
                if attempt + 1 < VARIANT.LLM_RETRY_ATTEMPTS:
                    await asyncio.sleep(
                        VARIANT.LLM_RETRY_JITTER_SECONDS
                        * (0.5 + random.random())
                    )

        logger.warning(
            "llm_verify_failed",
            product_a=str(a.id), product_b=str(b.id),
            error=str(last_exc)[:200] if last_exc else "unknown",
        )
        return None

    # ═══════════════════════════════════════════════════════
    # Targeted price recompute
    # ═══════════════════════════════════════════════════════
    async def _recompute_affected_prices(
        self, product_ids: list[UUID],
    ) -> None:
        if not product_ids:
            return
        rows = (await self.db.execute(
            text(
                "SELECT DISTINCT canonical_id FROM products "
                "WHERE id = ANY(CAST(:ids AS uuid[])) "
                "AND canonical_id IS NOT NULL"
            ),
            {"ids": [str(x) for x in product_ids]},
        )).all()
        canonical_ids = [r.canonical_id for r in rows]
        if canonical_ids:
            await recompute_canonical_cheapest(
                self.db, canonical_ids=canonical_ids,
            )
            await recompute_base_cheapest(self.db)

    # ═══════════════════════════════════════════════════════
    # Counter recompute for targeted runs
    # ═══════════════════════════════════════════════════════
    async def _recompute_counters_for_canonicals(
        self, canonical_ids: list[UUID],
    ) -> None:
        """
        Recompute listing_count, site_count on canonical_products
        and variant_count on the parent base_groups for the given
        canonical IDs. Keeps integrity checks green after a
        targeted rebuild.
        """
        if not canonical_ids:
            return

        await self.db.execute(
            text("""
                UPDATE canonical_products cp
                SET listing_count = sub.listing_count,
                    site_count = sub.site_count
                FROM (
                    SELECT
                        p.canonical_id,
                        COUNT(*) AS listing_count,
                        COUNT(DISTINCT p.site) AS site_count
                    FROM products p
                    WHERE p.canonical_id = ANY(CAST(:ids AS uuid[]))
                    GROUP BY p.canonical_id
                ) sub
                WHERE cp.id = sub.canonical_id
            """),
            {"ids": [str(x) for x in canonical_ids]},
        )

        base_ids = (await self.db.execute(
            text("""
                SELECT DISTINCT base_group_id
                FROM canonical_products
                WHERE id = ANY(CAST(:ids AS uuid[]))
                  AND base_group_id IS NOT NULL
            """),
            {"ids": [str(x) for x in canonical_ids]},
        )).scalars().all()

        if base_ids:
            await self.db.execute(
                text("""
                    UPDATE base_groups bg
                    SET variant_count = sub.variant_count
                    FROM (
                        SELECT
                            cp.base_group_id,
                            COUNT(*) AS variant_count
                        FROM canonical_products cp
                        WHERE cp.base_group_id = ANY(CAST(:bids AS uuid[]))
                        GROUP BY cp.base_group_id
                    ) sub
                    WHERE bg.id = sub.base_group_id
                """),
                {"bids": [str(x) for x in base_ids]},
            )

    # ═══════════════════════════════════════════════════════
    # Integrity check
    # ═══════════════════════════════════════════════════════
    async def _validate_integrity(self) -> int:
        errors = 0
        v = await self.db.execute(text(
            "SELECT COUNT(*) FROM canonical_products "
            "WHERE base_group_id IS NULL AND listing_count > 0"
        ))
        errors += int(v.scalar_one())

        v = await self.db.execute(text(
            "SELECT COUNT(*) FROM canonical_products cp "
            "WHERE cp.listing_count <> ("
            "  SELECT COUNT(*) FROM products p "
            "  WHERE p.canonical_id = cp.id"
            ")"
        ))
        errors += int(v.scalar_one())

        v = await self.db.execute(text(
            "SELECT COUNT(*) FROM base_groups bg "
            "WHERE bg.variant_count <> ("
            "  SELECT COUNT(*) FROM canonical_products cp "
            "  WHERE cp.base_group_id = bg.id"
            ")"
        ))
        errors += int(v.scalar_one())
        return errors


# ═══════════════════════════════════════════════════════
# Module helpers
# ═══════════════════════════════════════════════════════
def normalize_site(value: Optional[str]) -> str:
    return " ".join((value or "").strip().lower().split())


def normalize_text(value: Optional[str]) -> str:
    from app.core.product_tokenizer import normalize
    return normalize(value)


def _most_common(
    values: list[Optional[str]], default: Optional[str],
) -> Optional[str]:
    cleaned = [str(v) for v in values if v]
    if not cleaned:
        return default
    counts = Counter(cleaned)
    return min(counts, key=lambda v: (-counts[v], v.lower()))


def _most_common_serializable(values: list):
    counts = Counter(repr(v) for v in values)
    key = min(counts, key=lambda v: (-counts[v], v))
    return next(v for v in values if repr(v) == key)