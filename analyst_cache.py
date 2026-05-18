"""
Analyst ratings cache — stores Yahoo Finance analyst data in Postgres
to dramatically reduce yfinance calls and avoid Yahoo's rate limits.

Cache TTL: 6 hours. Analyst ratings don't change intraday, so this is
more than fresh enough. After the first fetch per stock per ~quarter-day,
all subsequent Analyze clicks return instantly from cache.

Schema: analyst_ratings_cache
  - symbol       TEXT PRIMARY KEY      (uppercased NSE symbol)
  - cached_at    TIMESTAMPTZ NOT NULL  (when this record was last refreshed)
  - payload      JSONB NOT NULL        (full ratings dict — same shape as
                                         what _fetch_analyst_ratings returns)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

# How long a cached entry stays valid. Analyst ratings change at most daily,
# so 6 hours is conservative — even during the busiest broker action window,
# our data is at most 6h stale.
CACHE_TTL_SECONDS = 6 * 60 * 60


@contextmanager
def _conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    c = psycopg2.connect(DATABASE_URL)
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _ensure_schema() -> None:
    """Idempotent — safe to call on every request. Creates the cache table
    if missing. Adds an index on cached_at for any future cleanup queries."""
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analyst_ratings_cache (
                    symbol      TEXT        PRIMARY KEY,
                    cached_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    payload     JSONB       NOT NULL
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_analyst_cache_age
                ON analyst_ratings_cache (cached_at);
            """)


def get_cached(symbol: str) -> Optional[dict]:
    """
    Return cached analyst data if it exists AND is fresh (< 6h old).

    Returns:
      The cached ratings dict if fresh.
      None if no cache entry or entry is stale.
    """
    symbol = symbol.upper().strip()
    try:
        _ensure_schema()
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    "SELECT cached_at, payload FROM analyst_ratings_cache "
                    "WHERE symbol = %s",
                    (symbol,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                cached_at, payload = row
                age_seconds = (datetime.now(timezone.utc) - cached_at).total_seconds()
                if age_seconds > CACHE_TTL_SECONDS:
                    logger.info(
                        "Cache STALE for %s (%.0fh old) — will refetch",
                        symbol, age_seconds / 3600,
                    )
                    return None

                logger.info(
                    "Cache HIT for %s (%.0fmin old)",
                    symbol, age_seconds / 60,
                )
                # psycopg2 returns JSONB as dict already
                return payload if isinstance(payload, dict) else json.loads(payload)
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", symbol, e)
        return None


def save_cache(symbol: str, payload: dict) -> None:
    """
    Store/update analyst data for this symbol. Idempotent (upserts).

    Only saves payloads that actually contain data. We don't want to
    cache empty results — that would defeat the purpose if a successful
    fetch happens later within the TTL window.
    """
    symbol = symbol.upper().strip()

    # Don't cache empty results — if yfinance returned nothing useful,
    # we want the next request to try again rather than return cached nothing.
    if payload.get("data_source") == "unavailable":
        logger.debug("Skipping cache save for %s — empty payload", symbol)
        return

    try:
        _ensure_schema()
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute("""
                    INSERT INTO analyst_ratings_cache (symbol, cached_at, payload)
                    VALUES (%s, NOW(), %s)
                    ON CONFLICT (symbol)
                    DO UPDATE SET cached_at = NOW(), payload = EXCLUDED.payload;
                """, (symbol, json.dumps(payload)))
        logger.info("Cache SAVED for %s", symbol)
    except Exception as e:
        # Cache failures must NEVER break the user-facing endpoint.
        # If saving fails, we just log and move on — the data is still
        # returned to the user, we just lose the speedup next time.
        logger.warning("Cache save failed for %s: %s", symbol, e)
