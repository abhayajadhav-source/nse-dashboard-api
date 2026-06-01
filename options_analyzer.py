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

    # IV / Greeks (added 2026 — populated by Upstox response or BS fallback)
    # atm_iv: average of ATM call+put IV, decimal form (0.25 = 25%)
    # days_to_expiry: integer days from today to expiry_date
    # Each entry in top_strikes also gets call_iv, put_iv, call_*, put_* Greek fields.
    atm_iv: Optional[float] = None
    days_to_expiry: int = 0


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


def _extract_greeks_from_upstox(option_block: dict) -> dict:
    """
    Try to pull IV + Greeks directly from Upstox's response if present.
    Upstox /v2/option/chain may include an `option_greeks` sub-object
    with `iv`, `delta`, `gamma`, `theta`, `vega`. Field names sometimes
    vary by API version, so we accept multiple key variants.

    Returns dict with keys "iv", "delta", "gamma", "theta", "vega" —
    any field missing/null gets `None`. Caller decides whether to
    compute the missing pieces via Black-Scholes.
    """
    if not option_block:
        return {"iv": None, "delta": None, "gamma": None, "theta": None, "vega": None}
    g = option_block.get("option_greeks") or option_block.get("greeks") or {}

    def _safe_float(value):
        try:
            v = float(value)
            # Reject NaN / inf / sentinel zeros that aren't real
            if v != v or v == float("inf") or v == float("-inf"):
                return None
            return v
        except (TypeError, ValueError):
            return None

    return {
        "iv":     _safe_float(g.get("iv") or g.get("implied_volatility")),
        "delta":  _safe_float(g.get("delta")),
        "gamma":  _safe_float(g.get("gamma")),
        "theta":  _safe_float(g.get("theta")),
        "vega":   _safe_float(g.get("vega")),
    }


def _compute_days_to_expiry(expiry_str: str) -> int:
    """Parse an expiry string like '2026-06-30' into days-from-today (>=0)."""
    from datetime import date, datetime
    if not expiry_str:
        return 0
    try:
        # Accept either 'YYYY-MM-DD' or 'YYYY-MM-DDT...'
        d = datetime.strptime(expiry_str[:10], "%Y-%m-%d").date()
        delta = (d - date.today()).days
        return max(delta, 0)
    except Exception:
        return 0


def _resolve_iv_and_greeks(option_block: dict, spot: float, strike: float,
                            days_to_expiry: int, ltp: float,
                            opt_type: str) -> dict:
    """
    Best-effort IV + Greeks resolution.

    Order of preference:
      1. Use Upstox-provided IV + Greeks if available
      2. If Upstox provides IV only → compute Greeks via BS using that IV
      3. If neither → reverse-solve IV from LTP, then compute Greeks
      4. If LTP is 0 / negative / invalid → return all-None (no data)

    Returns: {"iv", "delta", "gamma", "theta", "vega"} — values may be None.
    """
    # Lazy import to avoid circular issues
    from greeks import implied_volatility, black_scholes_greeks

    upstox = _extract_greeks_from_upstox(option_block)

    # Some Upstox responses return IV in PERCENT (e.g., 25.5) instead of
    # decimal (0.255). Normalize: if IV > 5.0 we assume it's a percent.
    iv = upstox.get("iv")
    if iv is not None and iv > 5.0:
        iv = iv / 100.0

    # If Upstox returned all four Greeks, use them as-is (just ensure scale)
    if (iv is not None and upstox.get("delta") is not None
            and upstox.get("gamma") is not None
            and upstox.get("theta") is not None
            and upstox.get("vega")  is not None):
        return {
            "iv":    round(iv, 4),
            "delta": upstox["delta"],
            "gamma": upstox["gamma"],
            "theta": upstox["theta"],
            "vega":  upstox["vega"],
        }

    # Reverse-solve IV from LTP if not given
    if iv is None:
        if ltp <= 0 or days_to_expiry <= 0 or spot <= 0 or strike <= 0:
            # No way to compute IV → return all-None
            return {"iv": None, "delta": None, "gamma": None,
                    "theta": None, "vega": None}
        iv = implied_volatility(spot, strike, ltp, days_to_expiry, opt_type)
        if iv is None:
            return {"iv": None, "delta": None, "gamma": None,
                    "theta": None, "vega": None}

    # Compute Greeks via Black-Scholes using the IV
    greeks = black_scholes_greeks(spot, strike, days_to_expiry, iv, opt_type)
    return {
        "iv":    round(iv, 4),
        "delta": greeks["delta"],
        "gamma": greeks["gamma"],
        "theta": greeks["theta"],
        "vega":  greeks["vega"],
    }


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

    # Compute days-to-expiry once (used for Greeks fallback)
    days_to_expiry = _compute_days_to_expiry(expiry)
    data.days_to_expiry = days_to_expiry

    strike_summary = []
    strikes_with_ltp = 0

    for row in near_atm:
        strike = float(row.get("strike_price", 0))

        call_oi, call_prev_oi, call_volume, _, call_ltp = _extract_market_data(row.get("call_options"))
        put_oi,  put_prev_oi,  put_volume,  _, put_ltp  = _extract_market_data(row.get("put_options"))

        # IV + Greeks: prefer Upstox-provided values, fallback to BS computation
        call_iv_greeks = _resolve_iv_and_greeks(
            row.get("call_options"), underlying_price, strike,
            days_to_expiry, call_ltp, "call",
        )
        put_iv_greeks  = _resolve_iv_and_greeks(
            row.get("put_options"), underlying_price, strike,
            days_to_expiry, put_ltp, "put",
        )

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
            # IV + Greeks per side (None if unavailable)
            "call_iv":             call_iv_greeks.get("iv"),
            "call_delta":          call_iv_greeks.get("delta"),
            "call_gamma":          call_iv_greeks.get("gamma"),
            "call_theta":          call_iv_greeks.get("theta"),
            "call_vega":           call_iv_greeks.get("vega"),
            "put_iv":              put_iv_greeks.get("iv"),
            "put_delta":           put_iv_greeks.get("delta"),
            "put_gamma":           put_iv_greeks.get("gamma"),
            "put_theta":           put_iv_greeks.get("theta"),
            "put_vega":            put_iv_greeks.get("vega"),
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

    # ---- ATM IV: average call+put IV at the strike nearest to spot ----
    # Uses the full strike_summary (not just top OI) so we get a real ATM,
    # not whichever strike happened to have the most OI. Falls back to None
    # if no nearby strike has valid IV data.
    if strike_summary:
        from greeks import compute_atm_iv
        # Re-sort by proximity to spot for ATM finding
        proximity_sorted = sorted(strike_summary,
                                  key=lambda s: abs(s["strike"] - underlying_price))
        data.atm_iv = compute_atm_iv(proximity_sorted[:5], underlying_price,
                                      data.days_to_expiry)
        if data.atm_iv is not None:
            logger.info("ATM IV for %s: %.2f%% (DTE=%d)",
                        symbol, data.atm_iv * 100, data.days_to_expiry)

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
