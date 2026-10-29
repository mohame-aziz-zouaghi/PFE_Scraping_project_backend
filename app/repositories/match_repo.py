"""Repository for product_match_pairs — the pairwise audit trail only.

Reads/writes for canonical_products and base_groups live in
CanonicalRepo. This split avoids the schema-drift trap (Gap 12).
"""
from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product_match import ProductMatchPair


class MatchRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ═══════════════════════════════════════════════════════
    # Candidate discovery
    # ═══════════════════════════════════════════════════════
    async def find_candidate_pairs(
        self,
        *,
        model: str,
        max_distance: float,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """KNN over product_embeddings, cross-site only.

        Returns [{product_a_id, product_b_id, cosine_distance}, ...]
        with a < b to avoid duplicate pairs.
        """
        sql = text(
            """
            SELECT
                LEAST(pe1.product_id, pe2.product_id)    AS product_a_id,
                GREATEST(pe1.product_id, pe2.product_id) AS product_b_id,
                (pe1.embedding <=> pe2.embedding)        AS cosine_distance
            FROM product_embeddings pe1
            JOIN products p1 ON p1.id = pe1.product_id
            JOIN product_embeddings pe2
                ON pe2.model = pe1.model
               AND pe2.product_id > pe1.product_id
            JOIN products p2 ON p2.id = pe2.product_id
            WHERE pe1.model = :model
              AND p1.is_enabled = TRUE
              AND p2.is_enabled = TRUE
              AND p1.site <> p2.site
              AND (pe1.embedding <=> pe2.embedding) <= :max_distance
            ORDER BY cosine_distance ASC, product_a_id, product_b_id
            """
        )
        params: dict[str, Any] = {
            "model": model,
            "max_distance": max_distance,
        }
        if limit:
            sql = text(str(sql) + " LIMIT :limit")
            params["limit"] = limit
        rows = (await self.db.execute(sql, params)).all()
        return [
            {
                "product_a_id": r.product_a_id,
                "product_b_id": r.product_b_id,
                "cosine_distance": float(r.cosine_distance),
            }
            for r in rows
        ]

    # ═══════════════════════════════════════════════════════
    # Pair upsert / fetch
    # ═══════════════════════════════════════════════════════
    async def upsert_pair(
        self,
        *,
        product_a_id: UUID,
        product_b_id: UUID,
        group_id: Optional[UUID] = None,
        cosine_distance: float = 0.0,
        decision: str,
        decision_source: str,
        variant_match: Optional[bool] = None,
        token_overlap: Optional[float] = None,
        llm_reason: Optional[str] = None,
        llm_confidence: Optional[float] = None,
    ) -> None:
        # Normalize ordering so (a, b) is a stable key.
        a, b = sorted([product_a_id, product_b_id], key=str)
        stmt = pg_insert(ProductMatchPair).values(
            product_a_id=a,
            product_b_id=b,
            canonical_id=group_id,
            cosine_distance=cosine_distance,
            decision=decision,
            decision_source=decision_source,
            variant_match=variant_match,
            token_overlap=token_overlap,
            llm_reason=llm_reason,
            llm_confidence=llm_confidence,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_match_pair",
            set_={
                "canonical_id": stmt.excluded.canonical_id,
                "cosine_distance": stmt.excluded.cosine_distance,
                "decision": stmt.excluded.decision,
                "decision_source": stmt.excluded.decision_source,
                "variant_match": stmt.excluded.variant_match,
                "token_overlap": stmt.excluded.token_overlap,
                "llm_reason": stmt.excluded.llm_reason,
                "llm_confidence": stmt.excluded.llm_confidence,
            },
        )
        await self.db.execute(stmt)

    async def get_pair(
        self, product_a_id: UUID, product_b_id: UUID
    ) -> Optional[ProductMatchPair]:
        a, b = sorted([product_a_id, product_b_id], key=str)
        stmt = select(ProductMatchPair).where(
            ProductMatchPair.product_a_id == a,
            ProductMatchPair.product_b_id == b,
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def link_pair_to_group(
        self, pair_id: int, canonical_id: UUID
    ) -> None:
        """Point a pair at its accepted canonical.

        Writes to canonical_id — the correct column in the split schema
        (Gap 5).
        """
        pair = await self.db.get(ProductMatchPair, pair_id)
        if pair is None:
            return
        pair.canonical_id = canonical_id
        await self.db.flush()

    # ═══════════════════════════════════════════════════════
    # Stats
    # ═══════════════════════════════════════════════════════
    async def stats(self) -> dict:
        out: dict[str, int] = {}
        for label, sql in (
            ("pairs_total", "SELECT COUNT(*) FROM product_match_pairs"),
            ("pairs_accepted",
             "SELECT COUNT(*) FROM product_match_pairs "
             "WHERE decision = 'accepted'"),
            ("pairs_rejected",
             "SELECT COUNT(*) FROM product_match_pairs "
             "WHERE decision = 'rejected'"),
            ("pairs_deferred",
             "SELECT COUNT(*) FROM product_match_pairs "
             "WHERE decision = 'borderline_deferred'"),
            ("pairs_error",
             "SELECT COUNT(*) FROM product_match_pairs "
             "WHERE decision = 'error'"),
        ):
            v = await self.db.execute(text(sql))
            out[label] = int(v.scalar_one())
        return out




    async def find_candidate_pairs_for_products(
        self,
        *,
        model: str,
        product_ids: list[UUID],
        max_distance: float,
        ) -> list[dict]:
        """Same as find_candidate_pairs, but only among `product_ids`."""
        if not product_ids:
            return []
        sql = text("""
            SELECT
                pe1.product_id AS product_a_id,
                pe2.product_id AS product_b_id,
                1 - (pe1.embedding <=> pe2.embedding) AS similarity,
                (pe1.embedding <=> pe2.embedding) AS cosine_distance
            FROM product_embeddings pe1
            JOIN product_embeddings pe2
            ON pe2.model = pe1.model
            AND pe2.product_id <> pe1.product_id
            WHERE pe1.model = :model
            AND pe1.product_id = ANY(CAST(:ids AS uuid[]))
            AND pe2.product_id = ANY(CAST(:ids AS uuid[]))
            AND (pe1.embedding <=> pe2.embedding) <= :max_distance
            AND pe1.product_id < pe2.product_id  -- dedupe pairs
            ORDER BY cosine_distance ASC
        """)
        rows = (await self.db.execute(sql, {
            "model": model,
            "ids": [str(x) for x in product_ids],
            "max_distance": max_distance,
        })).all()
        return [
            {
                "product_a_id": r.product_a_id,
                "product_b_id": r.product_b_id,
                "similarity": float(r.similarity),
                "cosine_distance": float(r.cosine_distance),
            }
            for r in rows
        ]


    async def find_candidate_pairs_for_products(
        self,
        *,
        model: str,
        product_ids: list[UUID],
        max_distance: float,
    ) -> list[dict]:
        """
        Same as find_candidate_pairs, but only among the given
        product IDs.

        Used by targeted re-matching: after the verification loop
        detaches a set of products from a bad canonical, this query
        finds candidate pairs only among those products, so the
        rest of the catalog isn't scanned.

        Uses `product_id < product_id` to deduplicate (a, b) vs
        (b, a) pairs — otherwise every pair would appear twice.
        """
        if not product_ids:
            return []

        sql = text(
            """
            SELECT
                pe1.product_id AS product_a_id,
                pe2.product_id AS product_b_id,
                1 - (pe1.embedding <=> pe2.embedding) AS similarity,
                (pe1.embedding <=> pe2.embedding) AS cosine_distance
            FROM product_embeddings pe1
            JOIN product_embeddings pe2
              ON pe2.model = pe1.model
             AND pe2.product_id <> pe1.product_id
            WHERE pe1.model = :model
              AND pe1.product_id = ANY(CAST(:ids AS uuid[]))
              AND pe2.product_id = ANY(CAST(:ids AS uuid[]))
              AND (pe1.embedding <=> pe2.embedding) <= :max_distance
              AND pe1.product_id < pe2.product_id
            ORDER BY cosine_distance ASC
            """
        )

        rows = (await self.db.execute(sql, {
            "model": model,
            "ids": [str(x) for x in product_ids],
            "max_distance": max_distance,
        })).all()

        return [
            {
                "product_a_id": r.product_a_id,
                "product_b_id": r.product_b_id,
                "similarity": float(r.similarity),
                "cosine_distance": float(r.cosine_distance),
            }
            for r in rows
        ]