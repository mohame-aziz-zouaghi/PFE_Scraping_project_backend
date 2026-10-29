# app/models/types.py
"""
SQLAlchemy type for pgvector's `vector` column.

Without this, the ORM declares the column as TEXT and asyncpg
sends strings as varchar, which Postgres rejects with:

    DatatypeMismatchError: column "embedding" is of type vector
    but expression is of type character varying

Usage:

    from app.models.types import Vector

    embedding: Mapped[list[float]] = mapped_column(
        Vector(768), nullable=False
    )

Reads come back as `list[float]`. Writes accept any iterable of
floats — a numpy array, a list, a tuple.
"""
from __future__ import annotations

from typing import Iterable

from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType):
    """
    Minimal pgvector adapter. Only the pieces we need:

      • Bind: a Python list[float] → PostgreSQL literal
        `'[0.1, 0.2, ...]'::vector`
      • Result: `[0.1, 0.2, ...]` string → list[float]
    """

    cache_ok = True

    def __init__(self, dim: int):
        self.dim = dim

    # SQLAlchemy uses this when emitting the column type in
    # DDL and when casting in queries.
    def get_col_spec(self, **kw) -> str:
        return f"vector({self.dim})"

    # ── Bind: Python → Postgres ────────────────────────
    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                # Already a serialized vector; pass through.
                return value
            if not isinstance(value, Iterable):
                raise TypeError(
                    f"Vector expects an iterable of floats, "
                    f"got {type(value).__name__}"
                )
            return "[" + ",".join(
                _fmt(x) for x in value
            ) + "]"
        return process

    # ── Result: Postgres → Python ──────────────────────
    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                # pgvector returns "[0.1,0.2,...]"
                inner = value.strip().lstrip("[").rstrip("]")
                if not inner:
                    return []
                return [float(x) for x in inner.split(",")]
            # asyncpg sometimes hands back a memoryview for
            # binary format; decode to text then parse.
            try:
                s = value.decode() if isinstance(value, (bytes, memoryview)) else str(value)
                inner = s.strip().lstrip("[").rstrip("]")
                if not inner:
                    return []
                return [float(x) for x in inner.split(",")]
            except Exception:
                return value
        return process


def _fmt(x) -> str:
    """Format a float without scientific notation surprises."""
    f = float(x)
    # pgvector accepts standard float repr; use %.7g to keep
    # it short without losing precision for typical embeddings.
    return f"{f:.7g}"