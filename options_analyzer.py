"""
NSE F&O options chain analyzer — Upstox edition (CALIBRATED FOR INDIAN STOCKS).

Replaces the original NSE-direct scraper (which was getting blocked). Now
fetches from Upstox's official /v2/option/chain endpoint via the long-lived
Analytics Token.

Computes the same metrics as before:
  - Put-Call Ratio (PCR) by OI
  - Max Pain price (strike where option writers benefit most)
  - OI Buildup analysis (long/short buildup, unwinding, covering)
  - Top OI strikes (act as dynamic S/R) — now includes call_ltp + put_ltp
  - Composite directional signal (Strong Bullish ... Strong Bearish)

CALIBRATION NOTES (May 2026):
  - PCR thresholds adjusted for Indian stock options, which naturally
    run lower than US/index PCR. Stock PCR ~0.7 is "normal", not bearish.
  - Buildup classification now uses CHANGE IN CALL+PUT OI direction
    (not PCR as proxy) since price direction is unavailable here.
  - Composite score weights rebalanced to remove bearish bias.

Output dataclass `OptionChainData` is unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from upstox_client import (
    UpstoxError, fetch_option_chain, get_instrument_key, get_nearest_expiry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PCR thresholds — CALIBRATED FOR INDIAN STOCK OPTIONS (May 2026)
#
# Background: PCR_OI for individual NSE stocks typically runs in the
# 0.5 to 1.0 range under normal conditions. The "balanced" reading is
# closer to 0.7-0.8, not 1.0 (the latter is more typical of NIFTY index).
#
# Old US-style thresholds (1.30/1.00/0.80/0.60) caused most stocks to
# read as Bearish because PCR 0.85 isn't bearish in Indian markets — it's
# normal/neutral.
#
# New calibration: shift the entire band down to match the actual
# distribution of Indian single-stock PCR.
# ---------------------------------------------------------------------------
PCR_STRONG_BULLISH = 1.20   # was 1.30 — heavy put writing
PCR_BULLISH        = 0.95   # was 1.00 — moderate put bias
PCR_BEARISH        = 0.55   # was 0.80 — sustained call writing dominance
PCR_STRONG_BEARISH = 0.40   # was 0.60 — extreme call buildup

STRIKES_AROUND_ATM = 10


@dataclass
class OptionChainData:
    """Computed metrics from option chain analysis."""
    symbol: str
    underlying_price: float
    expiry_date: str

    total_call_oi:        int = 0
    total_put_oi:         int = 0
    total_call_volume:    int = 0
    total_put_volume:     int = 0
    total_call_oi_change: int = 0
    total_put_oi_change:  int = 0

    pcr_oi:     float = 0.0
    pcr_volume: float = 0.0
    max_pain:   float = 0.0

    highest_call_oi_strike: float = 0.0
    highest_put_oi_strike:  float = 0.0

    pcr_signal:         str = "Neutral"
    buildup_signal:     str = "Neutral"
    composite_signal:   str = "Neutral"
    composite_strength: int = 0

    top_strikes: list = field(default_factory=list)
    confirmations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _extract_market_data(option_block: dict) -> tuple[int, int, int, float, float]:
    md = (option_block or {}).get("market_data") or {}
    oi          = int(md.get("oi", 0) or 0)
    prev_oi     = int(md.get("prev_oi", 0) or 0)
    volume      = int(md.get("volume", 0) or 0)
    close_price = float(md.get("close_price", 0) or 0)
    ltp         = float(md.get("ltp", 0) or 0)
    if ltp <= 0 and close_price > 0:
        ltp = close_price
    return oi, prev_oi, volume, close_price, ltp


def _compute_max_pain(rows: list) -> float:
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
    """
    Contrarian PCR interpretation — high PCR = bullish (heavy put writing).
    Thresholds calibrated for Indian stock options (lower than NIFTY index).
    """
    if pcr_oi >= PCR_STRONG_BULLISH: return "Strong Bullish"
    if pcr_oi >= PCR_BULLISH:        return "Bullish"
    if pcr_oi >= PCR_BEARISH:        return "Neutral"
    if pcr_oi >= PCR_STRONG_BEARISH: return "Bearish"
    return "Strong Bearish"


def _classify_buildup(call_chg: int, put_chg: int) -> str:
    """
    Classify the day's OI buildup pattern WITHOUT using PCR as a proxy.

    Interpretation (call writers are bearish; put writers are bullish):
      - Put OI rising AND Call OI falling → strong bullish (Long Buildup)
      - Call OI rising AND Put OI falling → strong bearish (Short Buildup)
      - Both rising: dominant side wins
      - Both falling: dominant side determines unwinding/covering
      - Roughly balanced → Neutral

    KEY FIX FROM PREVIOUS VERSION: removed the PCR-based branch that
    was inverting the signal for normal Indian PCR values (~0.7-0.85),
    causing virtually every stock to read as Short Buildup.
    """
    # Both unchanged — no meaningful signal
    if abs(call_chg) < 1000 and abs(put_chg) < 1000:
        return "Neutral"

    # Clean directional signals
    if put_chg > 0 and call_chg < 0:
        return "Long Buildup"           # bullish
    if call_chg > 0 and put_chg < 0:
        return "Short Buildup"          # bearish

    # Both rising — magnitude wins
    if put_chg > 0 and call_chg > 0:
        if put_chg > call_chg * 1.5:
            return "Long Buildup"       # puts dominate writing → bullish
        if call_chg > put_chg * 1.5:
            return "Short Buildup"      # calls dominate writing → bearish
        return "Neutral"

    # Both falling — magnitude wins (with different label)
    if put_chg < 0 and call_chg < 0:
        if abs(put_chg) > abs(call_chg) * 1.5:
            return "Short Covering"     # bears exiting → mildly bullish
        if abs(call_chg) > abs(put_chg) * 1.5:
            return "Long Unwinding"     # bulls exiting → mildly bearish
        return "Neutral"

    return "Neutral"


def _compute_composite_signal(data: OptionChainData) -> tuple[str, int, list]:
    """
    Vote-based composite signal.

    KEY FIXES FROM PREVIOUS VERSION:
      - Max pain bias REDUCED (was contributing -1 to almost every uptrending
        stock). Now only triggers at extreme distances (>4%, not >2%).
      - Score-to-label mapping is now SYMMETRIC.
    """
    score = 0
    confirmations: list[str] = []

    # PCR vote
    if data.pcr_signal == "Strong Bullish":
        score += 2
        confirmations.append(f"PCR {data.pcr_oi:.2f} (strong bullish — heavy put writing)")
    elif data.pcr_signal == "Bullish":
        score += 1
        confirmations.append(f"PCR {data.pcr_oi:.2f} (bullish — moderate put bias)")
    elif data.pcr_signal == "Bearish":
        score -= 1
        confirmations.append(f"PCR {data.pcr_oi:.2f} (bearish — call writing dominance)")
    elif data.pcr_signal == "Strong Bearish":
        score -= 2
        confirmations.append(f"PCR {data.pcr_oi:.2f} (strong bearish — heavy call writing)")

    # OI buildup vote
    if data.buildup_signal == "Long Buildup":
        score += 2
        confirmations.append("Long buildup — fresh longs / put writers adding")
    elif data.buildup_signal == "Short Covering":
        score += 1
        confirmations.append("Short covering — bears exiting")
    elif data.buildup_signal == "Short Buildup":
        score -= 2
        confirmations.append("Short buildup — fresh shorts / call writers adding")
    elif data.buildup_signal == "Long Unwinding":
        score -= 1
        confirmations.append("Long unwinding — bulls exiting")

    # Max-pain vote — only fires at extreme distances (>4%)
    if data.max_pain > 0:
        pct_from_pain = ((data.underlying_price - data.max_pain) / data.max_pain) * 100
        if pct_from_pain > 4:
            score -= 1
            confirmations.append(
                f"Price ({data.underlying_price:.1f}) is {pct_from_pain:+.1f}% above max pain "
                f"({data.max_pain:.1f}) — option writers want pull-down toward expiry"
            )
        elif pct_from_pain < -4:
            score += 1
            confirmations.append(
                f"Price ({data.underlying_price:.1f}) is {pct_from_pain:+.1f}% below max pain "
                f"({data.max_pain:.1f}) — option writers want pull-up toward expiry"
            )

    # Score → label (SYMMETRIC mapping)
    if   score >= 4:  label, strength = "Strong Bullish",   5
    elif score >= 2:  label, strength = "Bullish",          4
    elif score == 1:  label, strength = "Slightly Bullish", 3
    elif score == 0:  label, strength = "Neutral",          2
    elif score == -1: label, strength = "Slightly Bearish", 3
    elif score >= -3: label, strength = "Bearish",          4
    else:             label, strength = "Strong Bearish",   5

    return label, strength, confirmations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_options_data(symbol: str, underlying_price: float) -> Optional[OptionChainData]:
    """Main entry point — same signature as before."""
    instrument_key = get_instrument_key(symbol)
    if not instrument_key:
        logger.info("No instrument_key for %s — not F&O eligible?", symbol)
        return None

    expiry = get_nearest_expiry(instrument_key)
    if not expiry:
        logger.info("No valid expiry found for %s", symbol)
        return None

    payload = fetch_option_chain(instrument_key, expiry)
    if not payload:
        return None

    rows = payload.get("data") or []
    if not rows:
        logger.info("Empty option chain for %s / %s", symbol, expiry)
        return None

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

    # ---- Max pain ----
    data.max_pain = _compute_max_pain(rows)

    # ---- Top strikes ----
    strike_summary.sort(key=lambda x: x["total_oi"], reverse=True)
    data.top_strikes = strike_summary[:8]

    # ---- Classify signals (buildup no longer needs PCR) ----
    data.pcr_signal     = _classify_pcr(data.pcr_oi)
    data.buildup_signal = _classify_buildup(
        data.total_call_oi_change, data.total_put_oi_change,
    )
    data.composite_signal, data.composite_strength, data.confirmations = (
        _compute_composite_signal(data)
    )

    # Diagnostic log — confirms the new logic in deployed environments
    logger.info(
        "Signals for %s: PCR=%.2f (%s) · CallDOI=%+d, PutDOI=%+d (%s) · "
        "MaxPain=%.0f (%+.1f%%) -> Composite: %s (%d/5)",
        symbol, data.pcr_oi, data.pcr_signal,
        data.total_call_oi_change, data.total_put_oi_change, data.buildup_signal,
        data.max_pain,
        ((data.underlying_price - data.max_pain) / data.max_pain * 100) if data.max_pain > 0 else 0,
        data.composite_signal, data.composite_strength,
    )

    return data
