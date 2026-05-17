"""
NSE F&O options chain analyzer.

Fetches live option chain data from NSE's public JSON endpoint and computes:
  - Put-Call Ratio (PCR) by OI
  - Max Pain price (strike where option writers benefit most)
  - OI Buildup analysis (long/short buildup, unwinding, covering)
  - Top OI strikes (act as dynamic S/R)
  - Composite directional signal

NSE blocks naive scrapers — we mimic a browser session by hitting their
homepage first to collect cookies, then requesting the option chain JSON.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# NSE endpoints — these have been stable for years but could change
NSE_HOME_URL          = "https://www.nseindia.com/"
NSE_OPTION_CHAIN_URL  = "https://www.nseindia.com/api/option-chain-equities"

# Browser-like headers — NSE rejects requests without these
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
    "Referer":         "https://www.nseindia.com/option-chain",
    "Connection":      "keep-alive",
}

# Tunable thresholds
PCR_STRONG_BULLISH = 1.30   # PCR above this = strong bullish (oversold puts)
PCR_BULLISH        = 1.00
PCR_BEARISH        = 0.80
PCR_STRONG_BEARISH = 0.60   # PCR below this = strong bearish

# How many strikes around ATM to consider for analysis
STRIKES_AROUND_ATM = 10

# Cache session for ~10 minutes to avoid re-warming on every request
_SESSION_CACHE: dict = {"session": None, "created_at": 0}
SESSION_TTL_SECONDS = 600


@dataclass
class OptionChainData:
    """Computed metrics from option chain analysis."""
    symbol: str
    underlying_price: float
    expiry_date: str

    # Aggregates across all strikes
    total_call_oi:     int = 0
    total_put_oi:      int = 0
    total_call_volume: int = 0
    total_put_volume:  int = 0
    total_call_oi_change: int = 0
    total_put_oi_change:  int = 0

    # Derived metrics
    pcr_oi:     float = 0.0  # Total Put OI / Total Call OI
    pcr_volume: float = 0.0
    max_pain:   float = 0.0  # Strike at which option writers benefit most

    # Top concentrations
    highest_call_oi_strike: float = 0.0  # Strike with most Call OI — likely resistance
    highest_put_oi_strike:  float = 0.0  # Strike with most Put OI — likely support

    # Signal interpretation
    pcr_signal:     str = "Neutral"  # Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish
    buildup_signal: str = "Neutral"  # Long Buildup / Short Buildup / Long Unwinding / Short Covering
    composite_signal:    str = "Neutral"
    composite_strength:  int = 0    # 0-5

    # Top 5 strikes table
    top_strikes: list = field(default_factory=list)

    # Confirmation reasons (human-readable)
    confirmations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session management — NSE requires a warmed-up session with cookies
# ---------------------------------------------------------------------------
def _get_session() -> requests.Session:
    """Return a cookie-laden session ready to hit NSE's API."""
    now = time.time()
    cached = _SESSION_CACHE.get("session")
    age    = now - _SESSION_CACHE.get("created_at", 0)

    if cached is not None and age < SESSION_TTL_SECONDS:
        return cached

    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    # Hit homepage first to get cookies — NSE rejects API calls without them
    try:
        session.get(NSE_HOME_URL, timeout=10)
        # Small delay to look human-ish (NSE has bot detection)
        time.sleep(0.5)
    except Exception as e:
        logger.warning("Session warmup failed: %s", e)

    _SESSION_CACHE["session"]    = session
    _SESSION_CACHE["created_at"] = now
    return session


def _fetch_option_chain_raw(symbol: str) -> Optional[dict]:
    """Fetch raw option chain JSON from NSE."""
    for attempt in range(3):
        try:
            session = _get_session()
            response = session.get(
                NSE_OPTION_CHAIN_URL,
                params={"symbol": symbol.upper()},
                timeout=15,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # Session expired — invalidate and retry
                _SESSION_CACHE["session"] = None
                _SESSION_CACHE["created_at"] = 0
                time.sleep(1)
                continue
            else:
                logger.warning("NSE returned %d for %s", response.status_code, symbol)
                _SESSION_CACHE["session"] = None
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            logger.warning("NSE request failed (attempt %d): %s", attempt + 1, e)
            _SESSION_CACHE["session"] = None
            time.sleep(2)
        except ValueError:
            logger.warning("NSE returned invalid JSON for %s", symbol)
            return None
    return None


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------
def _compute_max_pain(records: list, expiry: str) -> float:
    """
    Max Pain: the strike at which the total intrinsic value of all OPEN
    options (calls + puts) is minimised. This is the price where option
    WRITERS lose least money at expiry — and where price often gravitates.

    Methodology:
      For each candidate strike S:
        total_pain[S] = Σ(call_OI[K] × max(S - K, 0)) for all K
                     + Σ(put_OI[K]  × max(K - S, 0)) for all K
      max_pain = argmin(total_pain)
    """
    # Collect all (strike, call_oi, put_oi) tuples for the chosen expiry
    strikes_data = []
    for rec in records:
        if rec.get("expiryDate") != expiry:
            continue
        strike  = rec.get("strikePrice")
        call_oi = (rec.get("CE") or {}).get("openInterest", 0)
        put_oi  = (rec.get("PE") or {}).get("openInterest", 0)
        if strike is not None:
            strikes_data.append((float(strike), int(call_oi), int(put_oi)))

    if not strikes_data:
        return 0.0

    candidate_strikes = sorted({s for s, _, _ in strikes_data})

    min_pain      = float("inf")
    max_pain_strike = candidate_strikes[0]

    for candidate in candidate_strikes:
        total_pain = 0.0
        for strike, call_oi, put_oi in strikes_data:
            # Call writers lose if spot finishes ABOVE strike
            if candidate > strike:
                total_pain += call_oi * (candidate - strike)
            # Put writers lose if spot finishes BELOW strike
            if candidate < strike:
                total_pain += put_oi  * (strike - candidate)
        if total_pain < min_pain:
            min_pain        = total_pain
            max_pain_strike = candidate

    return max_pain_strike


def _classify_oi_buildup(
    call_oi_change: int, put_oi_change: int,
    underlying_pct_change: float,
) -> tuple[str, str]:
    """
    Classify the day's OI buildup pattern. Returns (label, explanation).

    Standard interpretations:
      Price ↑ + Call OI ↑ → Long buildup (bullish)
      Price ↑ + Put OI  ↓ → Short covering (bullish, but weaker)
      Price ↓ + Put OI  ↑ → Short buildup (bearish)
      Price ↓ + Call OI ↓ → Long unwinding (bearish, but weaker)
    """
    price_up   = underlying_pct_change > 0.3
    price_down = underlying_pct_change < -0.3

    call_oi_up    = call_oi_change > 0
    call_oi_down  = call_oi_change < 0
    put_oi_up     = put_oi_change > 0
    put_oi_down   = put_oi_change < 0

    if price_up and call_oi_up:
        return "Long Buildup",
        "Price rising with fresh long positions — bullish continuation likely"
    if price_up and put_oi_down:
        return "Short Covering",
        "Price rising as shorts cover — bullish but weaker conviction"
    if price_down and put_oi_up:
        return "Short Buildup",
        "Price falling with fresh short positions — bearish continuation likely"
    if price_down and call_oi_down:
        return "Long Unwinding",
        "Price falling as longs exit — bearish but weaker conviction"

    return "Neutral", "OI activity doesn't show clear directional buildup"


def _classify_pcr(pcr_oi: float) -> str:
    """Standard contrarian interpretation of PCR."""
    if pcr_oi >= PCR_STRONG_BULLISH:
        return "Strong Bullish"
    if pcr_oi >= PCR_BULLISH:
        return "Bullish"
    if pcr_oi >= PCR_BEARISH:
        return "Neutral"
    if pcr_oi >= PCR_STRONG_BEARISH:
        return "Bearish"
    return "Strong Bearish"


def _compute_composite_signal(data: OptionChainData) -> tuple[str, int, list]:
    """
    Composite signal — combines PCR, buildup, and price-vs-max-pain.

    Each signal contributes a +1 / -1 / 0 vote. Strong signals get ±2.
    Final composite based on net votes.
    """
    score = 0
    confirmations = []

    # PCR vote
    if data.pcr_signal == "Strong Bullish":
        score += 2
        confirmations.append(f"PCR {data.pcr_oi:.2f} (strong bullish — heavy put writing)")
    elif data.pcr_signal == "Bullish":
        score += 1
        confirmations.append(f"PCR {data.pcr_oi:.2f} (bullish)")
    elif data.pcr_signal == "Bearish":
        score -= 1
        confirmations.append(f"PCR {data.pcr_oi:.2f} (bearish)")
    elif data.pcr_signal == "Strong Bearish":
        score -= 2
        confirmations.append(f"PCR {data.pcr_oi:.2f} (strong bearish — heavy call writing)")

    # OI buildup vote
    if data.buildup_signal == "Long Buildup":
        score += 2
        confirmations.append("Long buildup — fresh longs adding")
    elif data.buildup_signal == "Short Covering":
        score += 1
        confirmations.append("Short covering — shorts exiting")
    elif data.buildup_signal == "Short Buildup":
        score -= 2
        confirmations.append("Short buildup — fresh shorts adding")
    elif data.buildup_signal == "Long Unwinding":
        score -= 1
        confirmations.append("Long unwinding — longs exiting")

    # Price vs max-pain vote (where is price relative to where writers want it?)
    if data.max_pain > 0:
        pct_from_pain = ((data.underlying_price - data.max_pain) / data.max_pain) * 100
        if pct_from_pain > 2:
            score -= 1
            confirmations.append(
                f"Price ({data.underlying_price:.1f}) is {pct_from_pain:+.1f}% above max pain "
                f"({data.max_pain:.1f}) — pull-down bias"
            )
        elif pct_from_pain < -2:
            score += 1
            confirmations.append(
                f"Price ({data.underlying_price:.1f}) is {pct_from_pain:+.1f}% below max pain "
                f"({data.max_pain:.1f}) — pull-up bias"
            )

    # Map score to label
    if   score >= 4: label, strength = "Strong Bullish", 5
    elif score >= 2: label, strength = "Bullish",        4
    elif score >= 1: label, strength = "Slightly Bullish", 3
    elif score == 0: label, strength = "Neutral",        2
    elif score >= -1:label, strength = "Slightly Bearish", 3
    elif score >= -3:label, strength = "Bearish",        4
    else:            label, strength = "Strong Bearish", 5

    return label, strength, confirmations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_options_data(symbol: str, underlying_price: float) -> Optional[OptionChainData]:
    """
    Main entry point.

    Args:
      symbol: NSE F&O symbol (e.g. "RELIANCE", "HDFCBANK")
      underlying_price: Current spot price (we use this rather than the LTP
                        from option chain because it's fresher)

    Returns:
      OptionChainData with all metrics, or None if not an F&O stock or fetch fails.
    """
    raw = _fetch_option_chain_raw(symbol)
    if raw is None:
        return None

    # NSE response structure: { "records": { "data": [...], "expiryDates": [...] }, ... }
    records_section = raw.get("records") or {}
    records         = records_section.get("data") or []
    expiry_dates    = records_section.get("expiryDates") or []
    underlying_value = records_section.get("underlyingValue")

    if not records or not expiry_dates:
        logger.info("No option chain data for %s (not an F&O stock?)", symbol)
        return None

    # Use nearest expiry — most liquid + most relevant for short-term signal
    current_expiry = expiry_dates[0]
    if underlying_value:
        underlying_price = float(underlying_value)

    data = OptionChainData(
        symbol=symbol.upper(),
        underlying_price=underlying_price,
        expiry_date=current_expiry,
    )

    # Aggregate across the strikes near ATM (where the action is)
    # We sort strikes by distance from ATM and pick the closest N
    strike_records = []
    for rec in records:
        if rec.get("expiryDate") != current_expiry:
            continue
        strike = rec.get("strikePrice")
        if strike is None:
            continue
        strike_records.append((float(strike), rec))

    strike_records.sort(key=lambda x: abs(x[0] - underlying_price))
    near_atm_strikes = strike_records[:STRIKES_AROUND_ATM * 2]   # both above and below

    # Find highest-OI call & put strikes (for S/R reference)
    highest_call_oi = 0
    highest_put_oi  = 0

    for strike, rec in near_atm_strikes:
        ce = rec.get("CE") or {}
        pe = rec.get("PE") or {}

        call_oi          = int(ce.get("openInterest", 0))
        put_oi           = int(pe.get("openInterest", 0))
        call_oi_change   = int(ce.get("changeinOpenInterest", 0))
        put_oi_change    = int(pe.get("changeinOpenInterest", 0))
        call_volume      = int(ce.get("totalTradedVolume", 0))
        put_volume       = int(pe.get("totalTradedVolume", 0))

        data.total_call_oi        += call_oi
        data.total_put_oi         += put_oi
        data.total_call_oi_change += call_oi_change
        data.total_put_oi_change  += put_oi_change
        data.total_call_volume    += call_volume
        data.total_put_volume     += put_volume

        if call_oi > highest_call_oi:
            highest_call_oi = call_oi
            data.highest_call_oi_strike = strike
        if put_oi > highest_put_oi:
            highest_put_oi = put_oi
            data.highest_put_oi_strike = strike

    # Compute PCR
    if data.total_call_oi > 0:
        data.pcr_oi     = data.total_put_oi / data.total_call_oi
    if data.total_call_volume > 0:
        data.pcr_volume = data.total_put_volume / data.total_call_volume

    # Max pain (across full expiry, not just near ATM)
    data.max_pain = _compute_max_pain(records, current_expiry)

    # Build top-5 strikes table (sorted by total OI desc)
    strike_summary = []
    for strike, rec in near_atm_strikes:
        ce = rec.get("CE") or {}
        pe = rec.get("PE") or {}
        call_oi = int(ce.get("openInterest", 0))
        put_oi  = int(pe.get("openInterest", 0))
        strike_summary.append({
            "strike":         strike,
            "call_oi":        call_oi,
            "put_oi":         put_oi,
            "call_oi_change": int(ce.get("changeinOpenInterest", 0)),
            "put_oi_change":  int(pe.get("changeinOpenInterest", 0)),
            "total_oi":       call_oi + put_oi,
            "distance_from_spot": strike - underlying_price,
        })
    strike_summary.sort(key=lambda x: x["total_oi"], reverse=True)
    data.top_strikes = strike_summary[:8]

    # Day's price change (we infer this from yfinance separately; here we
    # compute a rough proxy from option-chain underlying data)
    # The OI buildup classification needs price direction — caller passes it via underlying_price
    # but we don't have prev close in this scope. We'll classify in the dashboard side.
    # For now, classify based on call/put OI imbalance.
    call_chg = data.total_call_oi_change
    put_chg  = data.total_put_oi_change

    # Use a simple heuristic since we don't have day's pct change here.
    # Better classification happens in the buildup tag below.
    if put_chg > call_chg and put_chg > 0:
        data.buildup_signal = "Short Buildup" if data.pcr_oi < 1.0 else "Long Buildup"
    elif call_chg > put_chg and call_chg > 0:
        data.buildup_signal = "Long Buildup"  if data.pcr_oi > 1.0 else "Short Buildup"
    elif put_chg < 0 and call_chg < 0:
        data.buildup_signal = "Long Unwinding"
    elif put_chg < 0 and call_chg > 0:
        data.buildup_signal = "Long Buildup"
    elif put_chg > 0 and call_chg < 0:
        data.buildup_signal = "Short Covering"
    else:
        data.buildup_signal = "Neutral"

    # Signals
    data.pcr_signal = _classify_pcr(data.pcr_oi)
    data.composite_signal, data.composite_strength, data.confirmations = (
        _compute_composite_signal(data)
    )

    return data
