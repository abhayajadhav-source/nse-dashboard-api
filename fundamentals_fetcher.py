"""
Fundamentals fetcher — pulls PE, EPS, ROE, 3Y avg profit growth, Promoter holding %.

Design philosophy:
  - All 5 metrics from Yahoo Finance (.info + .income_stmt)
  - Postgres cache with 7-day TTL since fundamentals don't change daily
  - Never raises — always returns a Fundamentals object with None for missing fields

Why Promoter holding instead of FII:
  - FII per-stock data is only published quarterly (lagging)
  - NSE has no clean API for it; scraping is fragile
  - Promoter holding is a stronger signal anyway for Indian retail
    (high + stable promoter holding = insider confidence)
  - Yahoo Finance has it reliably for every NSE F&O stock

The 7-day TTL is intentional:
  - PE/EPS update at quarterly results, not daily — caching saves ~250 API calls/day
  - Promoter holdings disclosed quarterly — no point refreshing more than weekly
  - Reduces Yahoo rate-limit pressure
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import psycopg2
import yfinance as yf

from config import IST, YF_RETRIES
from stock_list import get_yf_symbol

logger = logging.getLogger(__name__)

CACHE_TTL_HOURS = 7 * 24   # 7 days
DATABASE_URL = os.getenv("DATABASE_URL", "")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Fundamentals:
    symbol: str
    pe_ratio:                Optional[float] = None
    eps:                     Optional[float] = None
    roe_pct:                 Optional[float] = None    # already in %, not decimal
    profit_growth_3y_pct:    Optional[float] = None    # CAGR %, can be negative
    promoter_holding_pct:    Optional[float] = None    # % held by insiders/promoters
    fetched_at:              Optional[str]   = None    # ISO string


# ---------------------------------------------------------------------------
# Postgres cache
# ---------------------------------------------------------------------------
def _ensure_cache_table() -> None:
    """
    Create the fundamentals_cache table if it doesn't exist.

    Note: if you previously deployed the FII version of this module, there will
    be an old `fii_holding_pct` column in this table. That's fine — it just
    sits unused. We ALTER TABLE to add the new column on first run.
    """
    if not DATABASE_URL:
        return
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fundamentals_cache (
                        symbol                VARCHAR(32) PRIMARY KEY,
                        pe_ratio              NUMERIC,
                        eps                   NUMERIC,
                        roe_pct               NUMERIC,
                        profit_growth_3y_pct  NUMERIC,
                        fetched_at            TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                """)
                # Add the new column if it doesn't exist (idempotent migration
                # from the previous FII-based schema)
                cur.execute("""
                    ALTER TABLE fundamentals_cache
                    ADD COLUMN IF NOT EXISTS promoter_holding_pct NUMERIC
                """)
    except Exception as e:
        logger.warning("Could not ensure fundamentals_cache table: %s", e)


def _cache_get(symbol: str) -> Optional[Fundamentals]:
    """Return cached Fundamentals if fresh, else None."""
    if not DATABASE_URL:
        return None
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pe_ratio, eps, roe_pct, profit_growth_3y_pct,
                           promoter_holding_pct, fetched_at
                    FROM fundamentals_cache
                    WHERE symbol = %s
                """, (symbol,))
                row = cur.fetchone()
                if not row:
                    return None
                fetched_at = row[5]
                age_hours = (datetime.now(IST) - fetched_at).total_seconds() / 3600
                if age_hours > CACHE_TTL_HOURS:
                    return None  # expired
                return Fundamentals(
                    symbol               = symbol,
                    pe_ratio             = float(row[0]) if row[0] is not None else None,
                    eps                  = float(row[1]) if row[1] is not None else None,
                    roe_pct              = float(row[2]) if row[2] is not None else None,
                    profit_growth_3y_pct = float(row[3]) if row[3] is not None else None,
                    promoter_holding_pct = float(row[4]) if row[4] is not None else None,
                    fetched_at           = fetched_at.isoformat(),
                )
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", symbol, e)
        return None


def _cache_set(f: Fundamentals) -> None:
    """Upsert a fundamentals record."""
    if not DATABASE_URL:
        return
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO fundamentals_cache
                        (symbol, pe_ratio, eps, roe_pct, profit_growth_3y_pct,
                         promoter_holding_pct, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO UPDATE SET
                        pe_ratio             = EXCLUDED.pe_ratio,
                        eps                  = EXCLUDED.eps,
                        roe_pct              = EXCLUDED.roe_pct,
                        profit_growth_3y_pct = EXCLUDED.profit_growth_3y_pct,
                        promoter_holding_pct = EXCLUDED.promoter_holding_pct,
                        fetched_at           = EXCLUDED.fetched_at
                """, (
                    f.symbol, f.pe_ratio, f.eps, f.roe_pct,
                    f.profit_growth_3y_pct, f.promoter_holding_pct,
                    datetime.now(IST),
                ))
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", f.symbol, e)


# ---------------------------------------------------------------------------
# Yahoo Finance — all 5 metrics
# ---------------------------------------------------------------------------
def _fetch_from_yahoo(symbol: str) -> dict:
    """
    Pull PE, EPS, ROE, promoter holding from .info, then compute 3-year profit
    growth from annual income statements.

    Returns dict with keys: pe_ratio, eps, roe_pct, profit_growth_3y_pct,
    promoter_holding_pct. Each may be None if Yahoo doesn't have that field.
    """
    out = {
        "pe_ratio":             None,
        "eps":                  None,
        "roe_pct":              None,
        "profit_growth_3y_pct": None,
        "promoter_holding_pct": None,
    }
    yf_symbol = get_yf_symbol(symbol)

    for attempt in range(YF_RETRIES + 1):
        try:
            ticker = yf.Ticker(yf_symbol)

            # --- .info call (fastest path) ---
            info = ticker.info or {}
            out["pe_ratio"] = info.get("trailingPE")
            out["eps"]      = info.get("trailingEps")

            roe = info.get("returnOnEquity")
            # Yahoo returns ROE as a decimal (e.g. 0.18 = 18%); we want %.
            if roe is not None:
                out["roe_pct"] = roe * 100 if abs(roe) < 5 else roe

            # Promoter holding = "% held by insiders" in Yahoo terminology.
            # For Indian stocks, Yahoo populates this with promoter group data.
            # Returned as a decimal (e.g. 0.50 = 50%); convert to %.
            promoter = info.get("heldPercentInsiders")
            if promoter is not None:
                out["promoter_holding_pct"] = promoter * 100 if abs(promoter) < 5 else promoter

            # --- 3-year profit growth — separate API call ---
            try:
                income = ticker.income_stmt
                if income is not None and not income.empty:
                    if "Net Income" in income.index:
                        net_income_series = income.loc["Net Income"].dropna()
                        # Need at least 3+ years of data for a 3Y CAGR
                        if len(net_income_series) >= 4:
                            latest = float(net_income_series.iloc[0])
                            earlier = float(net_income_series.iloc[3])
                            if earlier > 0 and latest > 0:
                                cagr = ((latest / earlier) ** (1 / 3) - 1) * 100
                                out["profit_growth_3y_pct"] = cagr
                            elif earlier > 0 and latest <= 0:
                                # Went from profit to loss — assign large negative
                                out["profit_growth_3y_pct"] = -100.0
                            # If earlier was negative, CAGR is mathematically meaningless;
                            # leave as None rather than report a misleading number.
            except Exception as e:
                logger.debug("Income statement fetch failed for %s: %s", symbol, e)

            return out

        except Exception as e:
            err_str = str(e).lower()
            if "too many requests" in err_str and attempt < YF_RETRIES:
                backoff = 5 * (attempt + 1)
                logger.info("Rate limited on fundamentals for %s; sleeping %ds", symbol, backoff)
                time.sleep(backoff)
                continue
            if attempt < YF_RETRIES:
                time.sleep(1 + attempt)
                continue
            logger.warning("Fundamentals fetch failed for %s: %s", symbol, e)
            return out
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_fundamentals(symbol: str, use_cache: bool = True) -> Fundamentals:
    """
    Fetch all 5 fundamentals for a symbol, using cache where possible.

    Always returns a Fundamentals object — fields may be None if data is
    unavailable. Never raises.

    Args:
      symbol:    NSE ticker (e.g. "RELIANCE")
      use_cache: If True, check Postgres cache first (7-day TTL)
    """
    if use_cache:
        cached = _cache_get(symbol)
        if cached is not None:
            return cached

    # Cache miss — fetch fresh
    y = _fetch_from_yahoo(symbol)

    fundamentals = Fundamentals(
        symbol               = symbol,
        pe_ratio             = y.get("pe_ratio"),
        eps                  = y.get("eps"),
        roe_pct              = y.get("roe_pct"),
        profit_growth_3y_pct = y.get("profit_growth_3y_pct"),
        promoter_holding_pct = y.get("promoter_holding_pct"),
        fetched_at           = datetime.now(IST).isoformat(),
    )

    # Save to cache (even if some fields are None — we don't want to refetch
    # the same dead fields every run for 7 days)
    _cache_set(fundamentals)
    return fundamentals


# Ensure table exists on module import (no-op if already present)
_ensure_cache_table()
