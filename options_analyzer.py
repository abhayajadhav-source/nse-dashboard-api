"""
NSE F&O options chain analyzer — Upstox edition.

Replaces the original NSE-direct scraper (which was getting blocked). Now
fetches from Upstox's official /v2/option/chain endpoint via the long-lived
Analytics Token.

Computes the same metrics as before:
  - Put-Call Ratio (PCR) by OI
  - Max Pain price (strike where option writers benefit most)
  - OI Buildup analysis (long/short buildup, unwinding, covering)
  - Top OI strikes (act as dynamic S/R) — now includes call_ltp + put_ltp
    so the Strategy Advisor can price multi-leg strategies without
    refetching the chain.
  - Composite directional signal (Strong Bullish ... Strong Bearish)

Output dataclass `OptionChainData` is unchanged — keeps the Flask API,
AI prompt builder, and dashboard renderer compatible without changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from upstox_client import (
    UpstoxError, fetch_option_chain, get_instrument_key, get_nearest_expiry,
)

logger = logging.getLogger(__name__)

# Tunable thresholds — same as before
PCR_STRONG_BULLISH = 1.30
PCR_BULLISH        = 1.00
PCR_BEARISH        = 0.80
PCR_STRONG_BEARISH = 0.60

# How many strikes around ATM to consider for analysis
STRIKES_AROUND_ATM = 10


@dataclass
class OptionChainData:
    """Computed metrics from option chain analysis.

    SAME SHAPE as the previous NSE-based version — downstream callers
    (Flask app, AI prompt, dashboard) don't need to change.
    """
    symbol: str
    underlying_price: float
    expiry_date: str

    # Aggregates across all strikes
    total_call_oi:        int = 0
    total_put_oi:         int = 0
    total_call_volume:    int = 0
    total_put_volume:     int = 0
    total_call_oi_change: int = 0
    total_put_oi_change:  int = 0

    # Derived metrics
    pcr_oi:     float = 0.0
    pcr_volume: float = 0.0
    max_pain:   float = 0.0

    # Top concentrations
    highest_call_oi_strike: float = 0.0   # likely resistance
    highest_put_oi_strike:  float = 0.0   # likely support

    # Signal interpretation
    pcr_signal:         str = "Neutral"
    buildup_signal:     str = "Neutral"
    composite_signal:   str = "Neutral"
    composite_strength: int = 0    # 0-5

    # Top strikes table — list of dicts.
    # Each dict now includes call_ltp and put_ltp so the strategy engine
    # can price spreads/straddles without refetching the option chain.
    top_strikes: list = field(default_factory=list)

    # Human-readable confirmations
    confirmations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers — parse Upstox's option chain structure
# ---------------------------------------------------------------------------
def _extract_market_data(option_block: dict) -> tuple[int, int, int, float, float]:
    """
    Extract (OI, prev_OI, volume, close_price, ltp) from an option block.

    The Upstox response nests data under `market_data`:
      { "market_data": { "oi": N, "prev_oi": M, "volume": V,
                         "close_price": C, "ltp": L } }

    NOTE: We return `ltp` as a separate field. When market is closed, Upstox
    populates `close_price` but `ltp` may be 0; we fall back to close_price
    in that case so strategies always have a usable premium.
    """
    md = (option_block or {}).get("market_data") or {}
    oi          = int(md.get("oi", 0) or 0)
    prev_oi     = int(md.get("prev_oi", 0) or 0)
    volume      = int(md.get("volume", 0) or 0)
    close_price = float(md.get("close_price", 0) or 0)
    ltp         = float(md.get("ltp", 0) or 0)
    # Fall back to close_price if LTP is zero (e.g., after-market hours)
    if ltp <= 0 and close_price > 0:
        ltp = close_price
    return oi, prev_oi, volume, close_price, ltp


def _compute_max_pain(rows: list) -> float:
    """
    Max Pain: the strike where option WRITERS lose the least at expiry.

    total_pain[S] = Σ(call_OI[K] × max(S - K, 0)) for all K
                 + Σ(put_OI[K]  × max(K - S, 0)) for all K
    max_pain = argmin(total_pain)
    """
    strikes_data = []
    for row in rows:
        strike  = row.get("strike_price")
        call_oi, _, _, _, _ = _extract_market_data(row.get("call_options"))
        put_oi,  _, _, _, _ = _extract_market_data(row.get("put_options"))
        if strike is None:
            continue
        strikes_data.append((float(strike), call_oi, put_oi))

    if not strikes_data:
        return 0.0

    candidate_strikes = sorted({s for s, _, _ in strikes_data})

    min_pain        = float("inf")
    max_pain_strike = candidate_strikes[0]

    for candidate in candidate_strikes:
        total_pain = 0.0
        for strike, call_oi, put_oi in strikes_data:
            if candidate > strike:
                total_pain += call_oi * (candidate - strike)
            if candidate < strike:
                total_pain += put_oi  * (strike - candidate)
        if total_pain < min_pain:
            min_pain        = total_pain
            max_pain_strike = candidate

    return max_pain_strike


def _classify_pcr(pcr_oi: float) -> str:
    """Contrarian PCR interpretation — high PCR = bullish (heavy put writing)."""
    if pcr_oi >= PCR_STRONG_BULLISH: return "Strong Bullish"
    if pcr_oi >= PCR_BULLISH:        return "Bullish"
    if pcr_oi >= PCR_BEARISH:        return "Neutral"
    if pcr_oi >= PCR_STRONG_BEARISH: return "Bearish"
    return "Strong Bearish"


def _classify_buildup(call_chg: int, put_chg: int, pcr_oi: float) -> str:
    """
    Classify the day's OI buildup pattern.

    Simplified heuristic since we don't have a stable price-change proxy
    here. The full price-vs-OI matrix is:
      Price↑ + Call OI↑ → Long Buildup
      Price↑ + Put OI↓  → Short Covering
      Price↓ + Put OI↑  → Short Buildup
      Price↓ + Call OI↓ → Long Unwinding

    Since we don't have day's pct change available in this scope, we
    use the magnitude/direction of call vs put OI change to infer
    the dominant force.
    """
    if put_chg > call_chg and put_chg > 0:
        return "Short Buildup" if pcr_oi < 1.0 else "Long Buildup"
    if call_chg > put_chg and call_chg > 0:
        return "Long Buildup" if pcr_oi > 1.0 else "Short Buildup"
    if put_chg < 0 and call_chg < 0:
        return "Long Unwinding"
    if put_chg < 0 < call_chg:
        return "Long Buildup"
    if call_chg < 0 < put_chg:
        return "Short Covering"
    return "Neutral"


def _compute_composite_signal(data: OptionChainData) -> tuple[str, int, list]:
    """Vote-based composite signal — same logic as the NSE version."""
    score = 0
    confirmations: list[str] = []

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

    # Price vs max-pain vote
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

    # Score → label
    if   score >= 4:  label, strength = "Strong Bullish",   5
    elif score >= 2:  label, strength = "Bullish",          4
    elif score >= 1:  label, strength = "Slightly Bullish", 3
    elif score == 0:  label, strength = "Neutral",          2
    elif score >= -1: label, strength = "Slightly Bearish", 3
    elif score >= -3: label, strength = "Bearish",          4
    else:             label, strength = "Strong Bearish",   5

    return label, strength, confirmations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_options_data(symbol: str, underlying_price: float) -> Optional[OptionChainData]:
    """
    Main entry point — same signature as the old NSE version.

    Args:
      symbol:           NSE F&O symbol (e.g. "RELIANCE", "HDFCBANK")
      underlying_price: Current spot price from yfinance (used as fallback if
                        Upstox doesn't include it, and to compute distances)

    Returns:
      OptionChainData with all metrics, or None if not an F&O stock or fetch fails.
    """
    # Step 1: Resolve symbol → instrument_key
    instrument_key = get_instrument_key(symbol)
    if not instrument_key:
        logger.info("No instrument_key for %s — not F&O eligible?", symbol)
        return None

    # Step 2: Find the nearest valid expiry
    expiry = get_nearest_expiry(instrument_key)
    if not expiry:
        logger.info("No valid expiry found for %s", symbol)
        return None

    # Step 3: Fetch the full option chain
    payload = fetch_option_chain(instrument_key, expiry)
    if not payload:
        return None

    rows = payload.get("data") or []
    if not rows:
        logger.info("Empty option chain for %s / %s", symbol, expiry)
        return None

    # Upstox includes the underlying spot in each row — use that if available
    spot_from_payload = None
    for row in rows:
        spot = row.get("underlying_spot_price")
        if spot:
            spot_from_payload = float(spot)
            break
    if spot_from_payload:
        underlying_price = spot_from_payload

    data = OptionChainData(
        symbol           = symbol.upper(),
        underlying_price = underlying_price,
        expiry_date      = expiry,
    )

    # ---- Aggregate near-ATM strikes ----
    # Sort rows by distance from ATM, take the closest 2N
    rows_with_distance = []
    for row in rows:
        strike = row.get("strike_price")
        if strike is None:
            continue
        rows_with_distance.append((abs(float(strike) - underlying_price), row))
    rows_with_distance.sort(key=lambda x: x[0])
    near_atm = [r for _, r in rows_with_distance[: STRIKES_AROUND_ATM * 2]]

    highest_call_oi = 0
    highest_put_oi  = 0

    strike_summary = []
    # Track LTP coverage for diagnostic logging — if most strikes have zero
    # LTPs, the Strategy Advisor will produce no recommendations, so this
    # log line is critical for debugging.
    strikes_with_ltp = 0

    for row in near_atm:
        strike = float(row.get("strike_price", 0))

        call_oi, call_prev_oi, call_volume, _, call_ltp = _extract_market_data(row.get("call_options"))
        put_oi,  put_prev_oi,  put_volume,  _, put_ltp  = _extract_market_data(row.get("put_options"))

        call_oi_change = call_oi - call_prev_oi
        put_oi_change  = put_oi  - put_prev_oi

        data.total_call_oi        += call_oi
        data.total_put_oi         += put_oi
        data.total_call_oi_change += call_oi_change
        data.total_put_oi_change  += put_oi_change
        data.total_call_volume    += call_volume
        data.total_put_volume     += put_volume

        if call_oi > highest_call_oi:
            highest_call_oi             = call_oi
            data.highest_call_oi_strike = strike
        if put_oi > highest_put_oi:
            highest_put_oi             = put_oi
            data.highest_put_oi_strike = strike

        if call_ltp > 0 or put_ltp > 0:
            strikes_with_ltp += 1

        strike_summary.append({
            "strike":              strike,
            "call_oi":             call_oi,
            "put_oi":              put_oi,
            "call_oi_change":      call_oi_change,
            "put_oi_change":       put_oi_change,
            "total_oi":            call_oi + put_oi,
            "distance_from_spot":  strike - underlying_price,
            # NEW: include LTPs so Strategy Advisor can price multi-leg
            # strategies without refetching the chain. Defaults to 0.0 if
            # Upstox doesn't expose it; downstream strategy builders check
            # `if min(..., put_ltp) <= 0: return None` to skip unpriced legs.
            "call_ltp":            call_ltp,
            "put_ltp":             put_ltp,
        })

    logger.info(
        "Options chain parsed for %s: %d near-ATM strikes, %d with LTPs",
        symbol, len(strike_summary), strikes_with_ltp
    )

    # ---- PCR ----
    if data.total_call_oi > 0:
        data.pcr_oi = data.total_put_oi / data.total_call_oi
    if data.total_call_volume > 0:
        data.pcr_volume = data.total_put_volume / data.total_call_volume

    # Upstox also provides per-row PCR but we compute our own across near-ATM
    # for consistency with the previous NSE behaviour.

    # ---- Max pain (across the FULL chain, not just near-ATM) ----
    data.max_pain = _compute_max_pain(rows)

    # ---- Top strikes table (sorted by total OI desc) ----
    strike_summary.sort(key=lambda x: x["total_oi"], reverse=True)
    data.top_strikes = strike_summary[:8]

    # ---- Classify signals ----
    data.pcr_signal     = _classify_pcr(data.pcr_oi)
    data.buildup_signal = _classify_buildup(
        data.total_call_oi_change, data.total_put_oi_change, data.pcr_oi,
    )
    data.composite_signal, data.composite_strength, data.confirmations = (
        _compute_composite_signal(data)
    )

    return data
