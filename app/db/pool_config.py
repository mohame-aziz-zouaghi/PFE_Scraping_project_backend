"""
Pool configuration.

Two things the pool does:

    1. SIZE GROWS 5 -> 30 ON DEMAND
       Start at POOL_MIN_SIZE. Grow up to POOL_MAX_SIZE as concurrent
       leases demand it. Never grow past max, never shrink below min.

    2. IDLE CONNECTIONS EVICTED AFTER 5s
       Any connection unused for POOL_IDLE_TIMEOUT seconds is closed
       and returned to the OS. Shrinks down to POOL_MIN_SIZE.

All values overridable from environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class PoolConfig:
    # ── Pool sizing (5 -> 30) ──────────────────────────────────────────
    pool_min_size: int = field(
        default_factory=lambda: _env_int("POOL_MIN_SIZE", 5))
    pool_max_size: int = field(
        default_factory=lambda: _env_int("POOL_MAX_SIZE", 30))

    # ── Idle eviction (5s default) ─────────────────────────────────────
    pool_idle_timeout_s: float = field(
        default_factory=lambda: _env_float("POOL_IDLE_TIMEOUT_S", 120.0))

    # ── Evictor sweep interval (how often we check for idle conns) ─────
    pool_reaper_interval_s: float = field(
        default_factory=lambda: _env_float("POOL_REAPER_INTERVAL_S", 1.0))

    # ── How long a caller waits for a free slot ────────────────────────
    pool_acquire_timeout_s: float = field(
        default_factory=lambda: _env_float("POOL_ACQUIRE_TIMEOUT_S", 10.0))

    # ── Postgres statement timeouts ────────────────────────────────────
    db_statement_timeout_ms: int = field(
        default_factory=lambda: _env_int("DB_STATEMENT_TIMEOUT_MS", 120_000))
    db_idle_in_tx_timeout_ms: int = field(
        default_factory=lambda: _env_int("DB_IDLE_IN_TX_TIMEOUT_MS",0))

    # ── Slow-lease logging ─────────────────────────────────────────────
    db_lease_warn_ms: int = field(
        default_factory=lambda: _env_int("DB_LEASE_WARN_MS", 1000))
    redis_lease_warn_ms: int = field(
        default_factory=lambda: _env_int("REDIS_LEASE_WARN_MS", 500))

    # ── Redis op timeout (socket + grace) ──────────────────────────────
    redis_op_timeout_default: float = field(
        default_factory=lambda: _env_float("REDIS_OP_TIMEOUT", 7.0))

    def __post_init__(self) -> None:
        if self.pool_min_size > self.pool_max_size:
            raise ValueError(
                f"POOL_MIN_SIZE ({self.pool_min_size}) > "
                f"POOL_MAX_SIZE ({self.pool_max_size})"
            )
        if self.pool_min_size < 1:
            raise ValueError("POOL_MIN_SIZE must be >= 1")
        if self.pool_idle_timeout_s <= 0:
            raise ValueError("POOL_IDLE_TIMEOUT_S must be > 0")


POOL = PoolConfig()