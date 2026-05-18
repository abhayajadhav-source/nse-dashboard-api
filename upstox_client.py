"""
Upstox API client — single-purpose wrapper for option chain data.

Uses Upstox's Analytics Token (long-lived, read-only, 1-year validity).
Replaces the NSE direct scraping approach which was getting blocked.

Endpoints used:
  - GET /v2/instruments/search    → resolve symbol → instrument_key
  - GET /v2/option/chain          → fetch full option chain for an expiry

Auth:
  - Bearer token in Authorization header
  - Token set via UPSTOX_ACCESS_TOKEN env var
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

UPSTOX_BASE_URL          = "https://api.upstox.com"
INSTRUMENT_SEARCH_PATH   = "/v2/instruments/search"
OPTION_CHAIN_PATH        = "/v2/option/chain"

UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")

# In-memory cache for instrument_key lookups — symbols don't change for the
# lifetime of a process, so we never need to re-fetch them.
# Key: symbol string (e.g. "RELIANCE"), value: instrument_key (e.g. "NSE_EQ|INE002A01018")
_instrument_key_cache: dict[str, str] = {}

# How long a network call should wait before giving up
REQUEST_TIMEOUT_SECONDS = 15

# Retry config for transient failures
MAX_RETRIES         = 3
RETRY_BACKOFF_BASE  = 2.0   # exponential backoff: 2s, 4s, 8s


class UpstoxError(Exception):
    """Raised when Upstox API call fails after retries."""


class UpstoxAuthError(UpstoxError):
    """Raised when Upstox rejects our token (401)."""


class UpstoxNotFoundError(UpstoxError):
    """Raised when symbol/expiry doesn't exist (404)."""


# ---------------------------------------------------------------------------
# Low-level HTTP helper
# ---------------------------------------------------------------------------
def _headers() -> dict:
    """Build Upstox API headers. Centralised so token changes propagate."""
    if not UPSTOX_ACCESS_TOKEN:
        raise UpstoxError("UPSTOX_ACCESS_TOKEN env var is not set")
    return {
        "Accept":        "application/json",
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
    }


def _get_json(path: str, params: dict) -> dict:
    """
    GET request with retries and structured error handling.

    Returns:
      Parsed JSON dict on success.

    Raises:
      UpstoxAuthError    — token rejected (401)
      UpstoxNotFoundError — resource doesn't exist (404)
      UpstoxError        — other failures after retries
    """
    url = UPSTOX_BASE_URL + path

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                params=params,
                headers=_headers(),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            # Auth failures don't retry — token won't fix itself
            if response.status_code == 401:
                raise UpstoxAuthError(
                    "Upstox rejected token (401). Token expired or invalid. "
                    "Regenerate from account.upstox.com/developer/apps → Analytics."
                )

            # 404 = symbol/expiry not found — don't retry
            if response.status_code == 404:
                raise UpstoxNotFoundError(
                    f"Upstox returned 404 for {path} (params: {params}). "
                    "Symbol may not be F&O eligible or expiry date is invalid."
                )

            # Rate limit — back off and retry
            if response.status_code == 429:
                wait_seconds = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning("Upstox rate limited — waiting %.0fs", wait_seconds)
                time.sleep(wait_seconds)
                continue

            # Server errors — retry
            if response.status_code >= 500:
                wait_seconds = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Upstox %d on %s — retry in %.0fs",
                    response.status_code, path, wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            # Success path
            if response.status_code == 200:
                payload = response.json()
                # Upstox wraps responses as {"status": "success", "data": ...}
                # When status is "error", surface the message
                if payload.get("status") == "error":
                    errors = payload.get("errors", [])
                    msg = errors[0].get("message") if errors else "unknown error"
                    raise UpstoxError(f"Upstox API error: {msg}")
                return payload

            # Other unexpected status
            raise UpstoxError(
                f"Upstox {response.status_code} on {path}: {response.text[:200]}"
            )

        except requests.exceptions.Timeout as e:
            last_error = e
            wait_seconds = RETRY_BACKOFF_BASE * (2 ** attempt)
            logger.warning("Upstox timeout — retry %d in %.0fs", attempt + 1, wait_seconds)
            time.sleep(wait_seconds)
        except requests.exceptions.RequestException as e:
            last_error = e
            wait_seconds = RETRY_BACKOFF_BASE * (2 ** attempt)
            logger.warning("Upstox network error: %s — retry in %.0fs", e, wait_seconds)
            time.sleep(wait_seconds)

    raise UpstoxError(f"Upstox call failed after {MAX_RETRIES} retries: {last_error}")


# ---------------------------------------------------------------------------
# Instrument key resolution (Step 4 - Option B: live search via API)
# ---------------------------------------------------------------------------
def get_instrument_key(symbol: str) -> Optional[str]:
    """
    Resolve an NSE equity symbol to its Upstox instrument_key.

    Format returned: "NSE_EQ|INE...." (the equity ISIN-based key).
    This is the key needed for the /v2/option/chain endpoint.

    Caches results in memory — the instrument_key is stable per symbol
    for the lifetime of the underlying security.

    Args:
      symbol: NSE symbol (e.g. "RELIANCE", "HDFCBANK")

    Returns:
      instrument_key string, or None if not found / not F&O eligible.
    """
    symbol = symbol.upper().strip()

    # Fast path — already cached
    if symbol in _instrument_key_cache:
        return _instrument_key_cache[symbol]

    try:
        # Search for the equity. We want the EQ segment (the underlying),
        # not the FO segment (which would return individual option contracts).
        payload = _get_json(INSTRUMENT_SEARCH_PATH, {
            "query":            symbol,
            "exchanges":        "NSE",
            "segments":         "EQ",
            "page_number":      1,
            "records":          10,
        })
    except UpstoxNotFoundError:
        logger.info("No instrument found for symbol %s", symbol)
        return None
    except UpstoxError as e:
        logger.warning("Instrument search failed for %s: %s", symbol, e)
        return None

    # Response shape: {"status":"success", "data": [{"instrument_key": "NSE_EQ|...", "trading_symbol": "RELIANCE", ...}, ...]}
    candidates = payload.get("data", [])
    if not candidates:
        logger.info("No candidates returned for symbol %s", symbol)
        return None

    # Find the exact symbol match — search can return fuzzy matches
    for item in candidates:
        trading_symbol = (item.get("trading_symbol") or "").upper()
        if trading_symbol == symbol:
            instrument_key = item.get("instrument_key")
            if instrument_key:
                _instrument_key_cache[symbol] = instrument_key
                logger.info("Resolved %s → %s", symbol, instrument_key)
                return instrument_key

    # No exact match — take the first result and log a warning
    # (this handles symbols with corporate-action suffixes like RELIANCE-BE)
    first = candidates[0]
    instrument_key = first.get("instrument_key")
    if instrument_key:
        logger.warning(
            "No exact match for %s — using closest: %s (%s)",
            symbol, first.get("trading_symbol"), instrument_key,
        )
        _instrument_key_cache[symbol] = instrument_key
        return instrument_key

    return None


# ---------------------------------------------------------------------------
# Option chain fetch
# ---------------------------------------------------------------------------
def fetch_option_chain(instrument_key: str, expiry_date: str) -> Optional[dict]:
    """
    Fetch the full option chain for a given equity + expiry.

    Args:
      instrument_key: From get_instrument_key() — e.g. "NSE_EQ|INE002A01018"
      expiry_date:    "YYYY-MM-DD" — must match an actual NSE expiry date

    Returns:
      Raw Upstox option chain payload (dict), or None on failure.

    Response shape (Upstox /v2/option/chain):
      {
        "status": "success",
        "data": [
          {
            "expiry": "2026-05-29",
            "pcr": 0.93,
            "strike_price": 1200.0,
            "underlying_key": "NSE_EQ|INE...",
            "underlying_spot_price": 1245.50,
            "call_options": {
              "instrument_key": "NSE_FO|12345",
              "market_data": {
                "ltp": 52.5, "volume": 1234500, "oi": 4500, "prev_oi": 4200, "close_price": 50.0
              },
              "option_greeks": {
                "iv": 0.28, "delta": 0.65, "gamma": 0.003, "theta": -2.1, "vega": 8.5, "rho": 0.12
              }
            },
            "put_options": { ... same structure ... }
          },
          ...one entry per strike...
        ]
      }
    """
    try:
        payload = _get_json(OPTION_CHAIN_PATH, {
            "instrument_key": instrument_key,
            "expiry_date":    expiry_date,
        })
        return payload
    except UpstoxNotFoundError:
        logger.info("No option chain for %s on %s", instrument_key, expiry_date)
        return None
    except UpstoxError as e:
        logger.warning("Option chain fetch failed for %s/%s: %s",
                       instrument_key, expiry_date, e)
        return None


def get_nearest_expiry(instrument_key: str) -> Optional[str]:
    """
    Find the nearest upcoming expiry date for a given instrument's options.

    Strategy: try the current month's last Thursday, then the next month's.
    If neither works, walk forward week-by-week up to 6 weeks ahead.

    Returns the first expiry date (YYYY-MM-DD) that returns a valid option
    chain, or None if no valid expiry found.
    """
    from datetime import date, timedelta

    today = date.today()
    candidates: list[str] = []

    # Walk forward in weekly steps for ~6 weeks. NSE monthly expiries are
    # always the last Thursday of the month, but weekly expiries (Nifty/
    # Bank Nifty) are every Thursday. For stock options we mainly want
    # monthly, so we try Thursdays for the next 6 weeks.
    for week_offset in range(6):
        target = today + timedelta(days=week_offset * 7)
        # Find the next Thursday (weekday() == 3)
        days_to_thursday = (3 - target.weekday()) % 7
        thursday = target + timedelta(days=days_to_thursday)
        if thursday < today:
            continue
        candidate = thursday.strftime("%Y-%m-%d")
        if candidate not in candidates:
            candidates.append(candidate)

    # Try each candidate — first one that returns data wins
    for expiry_str in candidates:
        payload = fetch_option_chain(instrument_key, expiry_str)
        if payload and payload.get("data"):
            logger.info("Resolved nearest expiry for %s: %s", instrument_key, expiry_str)
            return expiry_str

    logger.warning("No valid expiry found for %s in next 6 weeks", instrument_key)
    return None
