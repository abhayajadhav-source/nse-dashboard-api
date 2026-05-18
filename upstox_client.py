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
OPTION_CONTRACT_PATH     = "/v2/option/contract"   # lists available expiries

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


def list_available_expiries(instrument_key: str) -> list[str]:
    """
    List all available option expiry dates for a given underlying.

    Uses Upstox's /v2/option/contract endpoint which returns every active
    option contract for the instrument. We extract the unique expiry dates,
    sort them ascending, and return as YYYY-MM-DD strings.

    This is the authoritative source — it doesn't matter whether NSE
    expiries are on Tuesday, Thursday, or any other day. Upstox tells us
    directly which dates are valid.

    Returns:
      List of expiry date strings in YYYY-MM-DD format, sorted nearest first.
      Empty list if the instrument has no F&O contracts.
    """
    try:
        payload = _get_json(OPTION_CONTRACT_PATH, {
            "instrument_key": instrument_key,
        })
    except UpstoxNotFoundError:
        return []
    except UpstoxError as e:
        logger.warning("Option contracts fetch failed for %s: %s", instrument_key, e)
        return []

    contracts = payload.get("data") or []
    if not contracts:
        return []

    # Each contract entry has an "expiry" field (Upstox returns either an
    # ISO date string or a Unix-millis timestamp — handle both).
    from datetime import date as _date, datetime as _datetime, timezone as _timezone

    unique_expiries: set[str] = set()
    for contract in contracts:
        raw_expiry = contract.get("expiry")
        if raw_expiry is None:
            continue
        # Normalize to YYYY-MM-DD
        try:
            if isinstance(raw_expiry, (int, float)):
                # Unix millis — use timezone-aware datetime
                dt = _datetime.fromtimestamp(raw_expiry / 1000, tz=_timezone.utc)
                expiry_str = dt.strftime("%Y-%m-%d")
            elif isinstance(raw_expiry, str):
                # Either "YYYY-MM-DD" or full ISO; take first 10 chars
                expiry_str = raw_expiry[:10]
            else:
                continue
            unique_expiries.add(expiry_str)
        except Exception as e:
            logger.debug("Could not parse expiry %r: %s", raw_expiry, e)
            continue

    # Filter to future expiries and sort ascending
    today = _date.today()
    future_expiries = []
    for s in unique_expiries:
        try:
            expiry_date = _datetime.strptime(s, "%Y-%m-%d").date()
            if expiry_date >= today:
                future_expiries.append(s)
        except ValueError:
            continue

    future_expiries.sort()
    return future_expiries


def get_nearest_expiry(instrument_key: str) -> Optional[str]:
    """
    Find the nearest upcoming option expiry for an instrument.

    Strategy (most reliable first):
      1. Ask Upstox /v2/option/contract for the list of active expiries
         and pick the nearest future one.
      2. If that fails (older Upstox account / API hiccup), fall back to
         day-by-day probing for the next 45 days. This handles any
         expiry-day rule (NSE stock options moved from Thursday to
         Tuesday in Sept 2025).

    Returns:
      Expiry date as YYYY-MM-DD, or None if nothing valid found.
    """
    # Strategy 1 — authoritative list from Upstox
    expiries = list_available_expiries(instrument_key)
    if expiries:
        nearest = expiries[0]
        logger.info("Resolved nearest expiry for %s: %s (from %d available)",
                    instrument_key, nearest, len(expiries))
        return nearest

    logger.info("Contracts endpoint returned no expiries for %s — falling back to day probing",
                instrument_key)

    # Strategy 2 — probe every weekday for the next 45 days
    # Covers Mon-Fri (skips weekends since exchange is closed) and works
    # regardless of whether expiries are on Tuesday, Thursday, or another day.
    from datetime import date, timedelta

    today = date.today()
    for day_offset in range(1, 46):
        target = today + timedelta(days=day_offset)
        if target.weekday() >= 5:   # 5 = Saturday, 6 = Sunday
            continue
        expiry_str = target.strftime("%Y-%m-%d")
        payload = fetch_option_chain(instrument_key, expiry_str)
        if payload and payload.get("data"):
            logger.info("Resolved nearest expiry via probing for %s: %s",
                        instrument_key, expiry_str)
            return expiry_str

    logger.warning("No valid expiry found for %s in next 45 days", instrument_key)
    return None
