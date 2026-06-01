"""
Chart Pattern Detection Engine for NSE stocks.

Detects two categories of patterns:

1. CANDLESTICK PATTERNS (single-bar or multi-bar rules)
   - Doji, Hammer, Hanging Man, Inverted Hammer, Shooting Star
   - Engulfing (bullish/bearish), Harami (bullish/bearish)
   - Morning Star, Evening Star
   - Three White Soldiers, Three Black Crows
   - Piercing Line, Dark Cloud Cover
   - Marubozu (bullish/bearish)

2. CHART PATTERNS (multi-day structure requiring peak/trough detection)
   - Double Top, Double Bottom
   - Triple Top, Triple Bottom
   - Head & Shoulders, Inverse Head & Shoulders

Each pattern is returned with:
   - type: 'bullish' | 'bearish' | 'neutral'
   - status: 'formed' (fully confirmed) | 'forming' (partial, needs more bars)
   - bars_ago: how many bars in the past the pattern occurred (0 = today)
   - reliability: 'high' | 'medium' | 'low'
   - description: human-readable explanation

Inputs:
   pandas DataFrame with columns: open, high, low, close, volume
   indexed by datetime, sorted ascending.

Outputs:
   list of dicts; empty list if nothing detected.

Reliability calibration:
   These patterns are imperfect signals. False-positive rate is REAL.
   Each pattern carries a reliability badge so the trader can weight it.
   Use patterns as context, not standalone entry signals.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np


# ===========================================================================
# Helpers
# ===========================================================================

def _body(row: pd.Series) -> float:
    """Absolute body size (close-open)."""
    return abs(row["close"] - row["open"])


def _range(row: pd.Series) -> float:
    """High-low range of the bar."""
    return row["high"] - row["low"]


def _upper_shadow(row: pd.Series) -> float:
    return row["high"] - max(row["open"], row["close"])


def _lower_shadow(row: pd.Series) -> float:
    return min(row["open"], row["close"]) - row["low"]


def _is_bullish(row: pd.Series) -> bool:
    return row["close"] > row["open"]


def _is_bearish(row: pd.Series) -> bool:
    return row["close"] < row["open"]


def _avg_body(df: pd.DataFrame, lookback: int = 14) -> float:
    """Average body size over recent bars — used to define 'long' candles."""
    if len(df) < lookback:
        return df.apply(_body, axis=1).mean()
    return df.iloc[-lookback:].apply(_body, axis=1).mean()


def _trend_before(df: pd.DataFrame, idx: int, lookback: int = 5) -> str:
    """
    Classify the trend leading up to bar `idx` (exclusive of `idx`).
    Uses simple linear regression slope on closes.
    Returns 'up', 'down', or 'sideways'.
    """
    start = max(0, idx - lookback)
    if idx - start < 3:
        return "sideways"
    closes = df["close"].iloc[start:idx].values
    x = np.arange(len(closes))
    # Slope as % of average price
    slope, intercept = np.polyfit(x, closes, 1)
    pct_slope = slope / closes.mean() * 100 if closes.mean() > 0 else 0
    if pct_slope > 0.3:
        return "up"
    elif pct_slope < -0.3:
        return "down"
    return "sideways"


# ===========================================================================
# Candlestick pattern detectors
# ===========================================================================
# Each detector returns either a dict (pattern found) or None.
# All operate on a window of recent bars (last 1-5 depending on the pattern).

def detect_doji(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Doji: body is tiny (<10% of range). Indicates indecision.
    Not directional on its own — reliability LOW unless context favors it.
    """
    if idx < 0 or idx >= len(df):
        return None
    row = df.iloc[idx]
    rng = _range(row)
    if rng <= 0:
        return None
    body_ratio = _body(row) / rng
    if body_ratio > 0.10:
        return None
    return {
        "name": "Doji",
        "type": "neutral",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "low",
        "description": "Indecision — open ≈ close. Watch the next bar for resolution.",
    }


def detect_hammer(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Hammer: small body near top, long lower shadow (>2× body),
    minimal upper shadow. Appears at bottom of downtrend → bullish reversal.
    """
    if idx < 0 or idx >= len(df):
        return None
    row = df.iloc[idx]
    body = _body(row)
    rng  = _range(row)
    if rng <= 0 or body <= 0:
        return None
    ls = _lower_shadow(row)
    us = _upper_shadow(row)
    if ls < 2 * body:           return None
    if us > body:                return None
    if body / rng > 0.35:        return None
    # Context: must follow a downtrend for "hammer" to be meaningful
    trend = _trend_before(df, idx)
    if trend != "down":
        return None
    return {
        "name": "Hammer",
        "type": "bullish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "medium",
        "description": "Long lower wick rejecting selloff — potential bullish reversal at bottom.",
    }


def detect_hanging_man(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """Hammer shape but at TOP of uptrend → bearish reversal warning."""
    if idx < 0 or idx >= len(df):
        return None
    row = df.iloc[idx]
    body = _body(row); rng = _range(row)
    if rng <= 0 or body <= 0:
        return None
    if _lower_shadow(row) < 2 * body:  return None
    if _upper_shadow(row) > body:       return None
    if body / rng > 0.35:               return None
    if _trend_before(df, idx) != "up":
        return None
    return {
        "name": "Hanging Man",
        "type": "bearish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "medium",
        "description": "Hammer shape after uptrend — selling pressure intraday, possible reversal.",
    }


def detect_shooting_star(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Shooting star: small body at bottom, long upper shadow, after uptrend.
    Bearish reversal signal.
    """
    if idx < 0 or idx >= len(df):
        return None
    row = df.iloc[idx]
    body = _body(row); rng = _range(row)
    if rng <= 0 or body <= 0:
        return None
    if _upper_shadow(row) < 2 * body:   return None
    if _lower_shadow(row) > body:       return None
    if body / rng > 0.35:               return None
    if _trend_before(df, idx) != "up":
        return None
    return {
        "name": "Shooting Star",
        "type": "bearish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "medium",
        "description": "Long upper wick rejecting rally — possible bearish reversal at top.",
    }


def detect_inverted_hammer(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """Shooting star shape but at BOTTOM of downtrend → bullish reversal."""
    if idx < 0 or idx >= len(df):
        return None
    row = df.iloc[idx]
    body = _body(row); rng = _range(row)
    if rng <= 0 or body <= 0:
        return None
    if _upper_shadow(row) < 2 * body:   return None
    if _lower_shadow(row) > body:       return None
    if body / rng > 0.35:               return None
    if _trend_before(df, idx) != "down":
        return None
    return {
        "name": "Inverted Hammer",
        "type": "bullish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "medium",
        "description": "Failed rally during downtrend — early bullish reversal sign, needs confirmation.",
    }


def detect_engulfing(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Engulfing: 2-bar pattern where current bar's body completely covers
    the previous body and is opposite direction. Reliable reversal signal.
    """
    if idx < 1 or idx >= len(df):
        return None
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    # Current bar must engulf the previous body
    curr_top = max(curr["open"], curr["close"])
    curr_bot = min(curr["open"], curr["close"])
    prev_top = max(prev["open"], prev["close"])
    prev_bot = min(prev["open"], prev["close"])
    if not (curr_top > prev_top and curr_bot < prev_bot):
        return None
    if _body(curr) < _body(prev) * 1.2:
        return None
    # Must be opposite direction
    if _is_bullish(curr) and _is_bearish(prev):
        if _trend_before(df, idx - 1) != "down":
            return None
        return {
            "name": "Bullish Engulfing",
            "type": "bullish",
            "status": "formed",
            "bars_ago": len(df) - 1 - idx,
            "reliability": "high",
            "description": "Strong bullish reversal — bull bar fully engulfs prior bear bar after downtrend.",
        }
    if _is_bearish(curr) and _is_bullish(prev):
        if _trend_before(df, idx - 1) != "up":
            return None
        return {
            "name": "Bearish Engulfing",
            "type": "bearish",
            "status": "formed",
            "bars_ago": len(df) - 1 - idx,
            "reliability": "high",
            "description": "Strong bearish reversal — bear bar fully engulfs prior bull bar after uptrend.",
        }
    return None


def detect_harami(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Harami: 2-bar pattern where current bar's body is INSIDE the previous body
    and opposite direction. Weaker reversal signal than engulfing.
    """
    if idx < 1 or idx >= len(df):
        return None
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    curr_top = max(curr["open"], curr["close"])
    curr_bot = min(curr["open"], curr["close"])
    prev_top = max(prev["open"], prev["close"])
    prev_bot = min(prev["open"], prev["close"])
    # Current body inside previous body
    if not (curr_top < prev_top and curr_bot > prev_bot):
        return None
    # Previous must be a meaningful body (not a doji)
    if _body(prev) < _avg_body(df.iloc[:idx]) * 0.8:
        return None
    if _is_bullish(curr) and _is_bearish(prev) and _trend_before(df, idx - 1) == "down":
        return {
            "name": "Bullish Harami",
            "type": "bullish",
            "status": "formed",
            "bars_ago": len(df) - 1 - idx,
            "reliability": "medium",
            "description": "Small bull bar inside prior bear bar — momentum slowing, possible reversal.",
        }
    if _is_bearish(curr) and _is_bullish(prev) and _trend_before(df, idx - 1) == "up":
        return {
            "name": "Bearish Harami",
            "type": "bearish",
            "status": "formed",
            "bars_ago": len(df) - 1 - idx,
            "reliability": "medium",
            "description": "Small bear bar inside prior bull bar — momentum slowing, possible reversal.",
        }
    return None


def detect_morning_star(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Morning Star: 3-bar bullish reversal.
    Bar 1: long bearish. Bar 2: small body (star). Bar 3: long bullish closing above
    midpoint of bar 1. Powerful bullish reversal.
    """
    if idx < 2 or idx >= len(df):
        return None
    b1 = df.iloc[idx - 2]
    b2 = df.iloc[idx - 1]
    b3 = df.iloc[idx]
    avg_body = _avg_body(df.iloc[:idx])
    if avg_body <= 0:
        return None
    # Bar 1: long bear
    if not (_is_bearish(b1) and _body(b1) > avg_body * 1.0):
        return None
    # Bar 2: small body (star)
    if _body(b2) > avg_body * 0.5:
        return None
    # Bar 2 opens with gap down from bar 1's body
    if max(b2["open"], b2["close"]) >= min(b1["open"], b1["close"]):
        return None
    # Bar 3: long bull, closes above midpoint of bar 1's body
    if not (_is_bullish(b3) and _body(b3) > avg_body * 1.0):
        return None
    midpoint = (b1["open"] + b1["close"]) / 2
    if b3["close"] < midpoint:
        return None
    if _trend_before(df, idx - 2) != "down":
        return None
    return {
        "name": "Morning Star",
        "type": "bullish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "high",
        "description": "Three-bar bullish reversal at downtrend bottom. Strong signal.",
    }


def detect_evening_star(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """Mirror of Morning Star — 3-bar bearish reversal at uptrend top."""
    if idx < 2 or idx >= len(df):
        return None
    b1 = df.iloc[idx - 2]
    b2 = df.iloc[idx - 1]
    b3 = df.iloc[idx]
    avg_body = _avg_body(df.iloc[:idx])
    if avg_body <= 0:
        return None
    if not (_is_bullish(b1) and _body(b1) > avg_body * 1.0):
        return None
    if _body(b2) > avg_body * 0.5:
        return None
    if min(b2["open"], b2["close"]) <= max(b1["open"], b1["close"]):
        return None
    if not (_is_bearish(b3) and _body(b3) > avg_body * 1.0):
        return None
    midpoint = (b1["open"] + b1["close"]) / 2
    if b3["close"] > midpoint:
        return None
    if _trend_before(df, idx - 2) != "up":
        return None
    return {
        "name": "Evening Star",
        "type": "bearish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "high",
        "description": "Three-bar bearish reversal at uptrend top. Strong signal.",
    }


def detect_three_white_soldiers(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Three consecutive long bullish bars, each closing higher.
    Strong bullish continuation/reversal signal.
    """
    if idx < 2 or idx >= len(df):
        return None
    b1, b2, b3 = df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]
    avg_body = _avg_body(df.iloc[:idx])
    if avg_body <= 0:
        return None
    if not (_is_bullish(b1) and _is_bullish(b2) and _is_bullish(b3)):
        return None
    if not (_body(b1) > avg_body * 0.8 and _body(b2) > avg_body * 0.8 and _body(b3) > avg_body * 0.8):
        return None
    # Each close higher than previous
    if not (b2["close"] > b1["close"] and b3["close"] > b2["close"]):
        return None
    # Opens within previous body (not gaps)
    if not (b1["open"] <= b2["open"] <= b1["close"]):
        return None
    if not (b2["open"] <= b3["open"] <= b2["close"]):
        return None
    return {
        "name": "Three White Soldiers",
        "type": "bullish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "high",
        "description": "Three strong consecutive bull bars — confirmed bullish momentum.",
    }


def detect_three_black_crows(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """Mirror of Three White Soldiers — bearish continuation/reversal."""
    if idx < 2 or idx >= len(df):
        return None
    b1, b2, b3 = df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]
    avg_body = _avg_body(df.iloc[:idx])
    if avg_body <= 0:
        return None
    if not (_is_bearish(b1) and _is_bearish(b2) and _is_bearish(b3)):
        return None
    if not (_body(b1) > avg_body * 0.8 and _body(b2) > avg_body * 0.8 and _body(b3) > avg_body * 0.8):
        return None
    if not (b2["close"] < b1["close"] and b3["close"] < b2["close"]):
        return None
    if not (b1["close"] <= b2["open"] <= b1["open"]):
        return None
    if not (b2["close"] <= b3["open"] <= b2["open"]):
        return None
    return {
        "name": "Three Black Crows",
        "type": "bearish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "high",
        "description": "Three strong consecutive bear bars — confirmed bearish momentum.",
    }


def detect_piercing_line(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    2-bar bullish reversal: long bear then long bull closing above midpoint
    of the prior bear's body.
    """
    if idx < 1 or idx >= len(df):
        return None
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    avg_body = _avg_body(df.iloc[:idx])
    if avg_body <= 0:
        return None
    if not (_is_bearish(prev) and _body(prev) > avg_body * 0.8):
        return None
    if not (_is_bullish(curr) and _body(curr) > avg_body * 0.8):
        return None
    # Curr opens below prev close (gap down), closes above midpoint of prev
    if curr["open"] >= prev["close"]:
        return None
    midpoint = (prev["open"] + prev["close"]) / 2
    if curr["close"] < midpoint:
        return None
    if curr["close"] > prev["open"]:
        # That would be engulfing, not piercing
        return None
    if _trend_before(df, idx - 1) != "down":
        return None
    return {
        "name": "Piercing Line",
        "type": "bullish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "medium",
        "description": "Bull bar pierces >50% into prior bear's body — bullish reversal.",
    }


def detect_dark_cloud_cover(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """Mirror of Piercing Line — 2-bar bearish reversal."""
    if idx < 1 or idx >= len(df):
        return None
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    avg_body = _avg_body(df.iloc[:idx])
    if avg_body <= 0:
        return None
    if not (_is_bullish(prev) and _body(prev) > avg_body * 0.8):
        return None
    if not (_is_bearish(curr) and _body(curr) > avg_body * 0.8):
        return None
    if curr["open"] <= prev["close"]:
        return None
    midpoint = (prev["open"] + prev["close"]) / 2
    if curr["close"] > midpoint:
        return None
    if curr["close"] < prev["open"]:
        return None
    if _trend_before(df, idx - 1) != "up":
        return None
    return {
        "name": "Dark Cloud Cover",
        "type": "bearish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "medium",
        "description": "Bear bar penetrates >50% into prior bull's body — bearish reversal.",
    }


def detect_marubozu(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Long body with no (or minimal) wicks — represents one-sided control of
    the entire session. Bullish or bearish depending on direction.
    """
    if idx < 0 or idx >= len(df):
        return None
    row = df.iloc[idx]
    body = _body(row); rng = _range(row)
    if rng <= 0 or body <= 0:
        return None
    avg_body = _avg_body(df.iloc[:idx + 1])
    if body < avg_body * 1.5:
        return None
    if body / rng < 0.95:
        return None
    if _is_bullish(row):
        return {
            "name": "Bullish Marubozu",
            "type": "bullish",
            "status": "formed",
            "bars_ago": len(df) - 1 - idx,
            "reliability": "medium",
            "description": "Long bull bar with no wicks — buyers fully in control.",
        }
    return {
        "name": "Bearish Marubozu",
        "type": "bearish",
        "status": "formed",
        "bars_ago": len(df) - 1 - idx,
        "reliability": "medium",
        "description": "Long bear bar with no wicks — sellers fully in control.",
    }


# ===========================================================================
# Chart pattern detectors (peak/trough based)
# ===========================================================================

def _find_peaks_and_troughs(df: pd.DataFrame, window: int = 5,
                             min_prominence_pct: float = 1.5) -> tuple[list, list]:
    """
    Find local peaks (highs) and troughs (lows) in the price series.
    A peak/trough at index i requires:
      - i's high (peak) or low (trough) is the extreme in [i-window, i+window]
      - prominence: peak/trough differs from surrounding by >= min_prominence_pct%

    Returns: (peaks_indices, troughs_indices)
    """
    if len(df) < 2 * window + 1:
        return [], []
    highs = df["high"].values
    lows  = df["low"].values
    closes = df["close"].values
    peaks = []
    troughs = []
    for i in range(window, len(df) - window):
        local_window_highs = highs[i - window : i + window + 1]
        local_window_lows  = lows[i - window : i + window + 1]
        # Peak check
        if highs[i] == local_window_highs.max():
            # Prominence: peak must be >X% above nearest neighbors
            ref_low = closes[max(0, i - window * 2) : i + 1].min()
            if ref_low > 0 and (highs[i] - ref_low) / ref_low * 100 >= min_prominence_pct:
                peaks.append(i)
        # Trough check
        if lows[i] == local_window_lows.min():
            ref_high = closes[max(0, i - window * 2) : i + 1].max()
            if ref_high > 0 and (ref_high - lows[i]) / ref_high * 100 >= min_prominence_pct:
                troughs.append(i)
    return peaks, troughs


def detect_double_top(df: pd.DataFrame, peaks: list, troughs: list) -> Optional[dict]:
    """
    Double Top: two peaks of roughly equal height separated by a valley.
    Bearish reversal. Confirmed when price closes below the neckline (valley).

    Forming: two peaks found but neckline not yet broken.
    Formed:  neckline broken after second peak.
    """
    if len(peaks) < 2 or len(troughs) < 1:
        return None
    # Use the two most recent peaks
    p2 = peaks[-1]
    p1 = peaks[-2]
    if p2 - p1 < 5:
        return None  # Too close — not really separate peaks
    # Find the lowest trough between them (neckline)
    between_troughs = [t for t in troughs if p1 < t < p2]
    if not between_troughs:
        return None
    neckline_idx = min(between_troughs, key=lambda t: df["low"].iloc[t])
    neckline = df["low"].iloc[neckline_idx]
    peak1_high = df["high"].iloc[p1]
    peak2_high = df["high"].iloc[p2]
    # Peaks must be roughly equal (within 3%)
    if abs(peak1_high - peak2_high) / max(peak1_high, peak2_high) > 0.03:
        return None
    # The current bar must be AFTER the second peak
    last_idx = len(df) - 1
    if last_idx - p2 > 20:
        return None  # Stale — pattern from too long ago
    # Status: formed if price has closed below neckline since p2; else forming
    closes_after_p2 = df["close"].iloc[p2 + 1 :].values
    formed = any(c < neckline for c in closes_after_p2)
    target = neckline - (peak1_high - neckline)  # measured move
    return {
        "name": "Double Top",
        "type": "bearish",
        "status": "formed" if formed else "forming",
        "bars_ago": last_idx - p2,
        "reliability": "high" if formed else "medium",
        "description": (
            f"Two peaks near ₹{peak1_high:.2f} / ₹{peak2_high:.2f}, "
            f"neckline at ₹{neckline:.2f}. "
            + ("Neckline broken — bearish confirmed." if formed else
               f"Watch for close below ₹{neckline:.2f} to confirm.")
        ),
        "neckline": round(neckline, 2),
        "target":   round(target, 2),
    }


def detect_double_bottom(df: pd.DataFrame, peaks: list, troughs: list) -> Optional[dict]:
    """Mirror of Double Top — bullish reversal."""
    if len(troughs) < 2 or len(peaks) < 1:
        return None
    t2 = troughs[-1]
    t1 = troughs[-2]
    if t2 - t1 < 5:
        return None
    between_peaks = [p for p in peaks if t1 < p < t2]
    if not between_peaks:
        return None
    neckline_idx = max(between_peaks, key=lambda p: df["high"].iloc[p])
    neckline = df["high"].iloc[neckline_idx]
    trough1_low = df["low"].iloc[t1]
    trough2_low = df["low"].iloc[t2]
    if abs(trough1_low - trough2_low) / max(trough1_low, trough2_low) > 0.03:
        return None
    last_idx = len(df) - 1
    if last_idx - t2 > 20:
        return None
    closes_after_t2 = df["close"].iloc[t2 + 1 :].values
    formed = any(c > neckline for c in closes_after_t2)
    target = neckline + (neckline - trough1_low)
    return {
        "name": "Double Bottom",
        "type": "bullish",
        "status": "formed" if formed else "forming",
        "bars_ago": last_idx - t2,
        "reliability": "high" if formed else "medium",
        "description": (
            f"Two troughs near ₹{trough1_low:.2f} / ₹{trough2_low:.2f}, "
            f"neckline at ₹{neckline:.2f}. "
            + ("Neckline broken — bullish confirmed." if formed else
               f"Watch for close above ₹{neckline:.2f} to confirm.")
        ),
        "neckline": round(neckline, 2),
        "target":   round(target, 2),
    }


def detect_head_and_shoulders(df: pd.DataFrame, peaks: list, troughs: list) -> Optional[dict]:
    """
    Head and Shoulders: 3 peaks where middle peak is highest (head),
    flanked by two roughly-equal shoulders. Bearish reversal.

    Forming: structure visible but neckline not broken.
    Formed: neckline broken after right shoulder.
    """
    if len(peaks) < 3 or len(troughs) < 2:
        return None
    # Take last 3 peaks
    p_left, p_head, p_right = peaks[-3], peaks[-2], peaks[-1]
    if p_right - p_left < 10:
        return None
    h_left  = df["high"].iloc[p_left]
    h_head  = df["high"].iloc[p_head]
    h_right = df["high"].iloc[p_right]
    # Head must be highest
    if not (h_head > h_left and h_head > h_right):
        return None
    # Shoulders within 5% of each other
    if abs(h_left - h_right) / max(h_left, h_right) > 0.05:
        return None
    # Head must be at least 2% above the shoulders
    if (h_head - max(h_left, h_right)) / h_head < 0.02:
        return None
    # Find two troughs between the peaks for neckline
    troughs_left_mid  = [t for t in troughs if p_left < t < p_head]
    troughs_mid_right = [t for t in troughs if p_head < t < p_right]
    if not troughs_left_mid or not troughs_mid_right:
        return None
    t1 = min(troughs_left_mid, key=lambda t: df["low"].iloc[t])
    t2 = min(troughs_mid_right, key=lambda t: df["low"].iloc[t])
    neckline = (df["low"].iloc[t1] + df["low"].iloc[t2]) / 2
    last_idx = len(df) - 1
    if last_idx - p_right > 20:
        return None
    closes_after_right = df["close"].iloc[p_right + 1 :].values
    formed = any(c < neckline for c in closes_after_right)
    target = neckline - (h_head - neckline)
    return {
        "name": "Head & Shoulders",
        "type": "bearish",
        "status": "formed" if formed else "forming",
        "bars_ago": last_idx - p_right,
        "reliability": "high" if formed else "medium",
        "description": (
            f"Head ₹{h_head:.2f}, shoulders ₹{h_left:.2f}/₹{h_right:.2f}, "
            f"neckline ₹{neckline:.2f}. "
            + ("Neckline broken — bearish confirmed." if formed else
               f"Watch close below ₹{neckline:.2f} to confirm.")
        ),
        "neckline": round(neckline, 2),
        "target":   round(target, 2),
    }


def detect_inverse_head_and_shoulders(df: pd.DataFrame, peaks: list, troughs: list) -> Optional[dict]:
    """Mirror of H&S — bullish reversal at downtrend bottom."""
    if len(troughs) < 3 or len(peaks) < 2:
        return None
    t_left, t_head, t_right = troughs[-3], troughs[-2], troughs[-1]
    if t_right - t_left < 10:
        return None
    l_left  = df["low"].iloc[t_left]
    l_head  = df["low"].iloc[t_head]
    l_right = df["low"].iloc[t_right]
    if not (l_head < l_left and l_head < l_right):
        return None
    if abs(l_left - l_right) / max(l_left, l_right) > 0.05:
        return None
    if (min(l_left, l_right) - l_head) / l_head < 0.02:
        return None
    peaks_left_mid  = [p for p in peaks if t_left < p < t_head]
    peaks_mid_right = [p for p in peaks if t_head < p < t_right]
    if not peaks_left_mid or not peaks_mid_right:
        return None
    p1 = max(peaks_left_mid, key=lambda p: df["high"].iloc[p])
    p2 = max(peaks_mid_right, key=lambda p: df["high"].iloc[p])
    neckline = (df["high"].iloc[p1] + df["high"].iloc[p2]) / 2
    last_idx = len(df) - 1
    if last_idx - t_right > 20:
        return None
    closes_after_right = df["close"].iloc[t_right + 1 :].values
    formed = any(c > neckline for c in closes_after_right)
    target = neckline + (neckline - l_head)
    return {
        "name": "Inverse Head & Shoulders",
        "type": "bullish",
        "status": "formed" if formed else "forming",
        "bars_ago": last_idx - t_right,
        "reliability": "high" if formed else "medium",
        "description": (
            f"Head ₹{l_head:.2f}, shoulders ₹{l_left:.2f}/₹{l_right:.2f}, "
            f"neckline ₹{neckline:.2f}. "
            + ("Neckline broken — bullish confirmed." if formed else
               f"Watch close above ₹{neckline:.2f} to confirm.")
        ),
        "neckline": round(neckline, 2),
        "target":   round(target, 2),
    }


def detect_triple_top(df: pd.DataFrame, peaks: list, troughs: list) -> Optional[dict]:
    """Three peaks of roughly equal height. Bearish reversal."""
    if len(peaks) < 3 or len(troughs) < 2:
        return None
    p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
    if p3 - p1 < 15:
        return None
    h1, h2, h3 = df["high"].iloc[p1], df["high"].iloc[p2], df["high"].iloc[p3]
    avg_h = (h1 + h2 + h3) / 3
    if any(abs(h - avg_h) / avg_h > 0.025 for h in [h1, h2, h3]):
        return None
    troughs_between = [t for t in troughs if p1 < t < p3]
    if len(troughs_between) < 2:
        return None
    neckline = min(df["low"].iloc[t] for t in troughs_between)
    last_idx = len(df) - 1
    if last_idx - p3 > 20:
        return None
    formed = any(c < neckline for c in df["close"].iloc[p3 + 1 :].values)
    target = neckline - (avg_h - neckline)
    return {
        "name": "Triple Top",
        "type": "bearish",
        "status": "formed" if formed else "forming",
        "bars_ago": last_idx - p3,
        "reliability": "high" if formed else "medium",
        "description": (
            f"Three peaks near ₹{avg_h:.2f}, neckline ₹{neckline:.2f}. "
            + ("Bearish confirmed." if formed else "Watch neckline for confirmation.")
        ),
        "neckline": round(neckline, 2),
        "target":   round(target, 2),
    }


def detect_triple_bottom(df: pd.DataFrame, peaks: list, troughs: list) -> Optional[dict]:
    """Mirror of Triple Top — bullish reversal."""
    if len(troughs) < 3 or len(peaks) < 2:
        return None
    t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
    if t3 - t1 < 15:
        return None
    l1, l2, l3 = df["low"].iloc[t1], df["low"].iloc[t2], df["low"].iloc[t3]
    avg_l = (l1 + l2 + l3) / 3
    if any(abs(l - avg_l) / avg_l > 0.025 for l in [l1, l2, l3]):
        return None
    peaks_between = [p for p in peaks if t1 < p < t3]
    if len(peaks_between) < 2:
        return None
    neckline = max(df["high"].iloc[p] for p in peaks_between)
    last_idx = len(df) - 1
    if last_idx - t3 > 20:
        return None
    formed = any(c > neckline for c in df["close"].iloc[t3 + 1 :].values)
    target = neckline + (neckline - avg_l)
    return {
        "name": "Triple Bottom",
        "type": "bullish",
        "status": "formed" if formed else "forming",
        "bars_ago": last_idx - t3,
        "reliability": "high" if formed else "medium",
        "description": (
            f"Three troughs near ₹{avg_l:.2f}, neckline ₹{neckline:.2f}. "
            + ("Bullish confirmed." if formed else "Watch neckline for confirmation.")
        ),
        "neckline": round(neckline, 2),
        "target":   round(target, 2),
    }


# ===========================================================================
# Master detection function
# ===========================================================================

# All candlestick detectors that work bar-by-bar at the latest bar
_CANDLESTICK_DETECTORS = [
    detect_engulfing,           # 2-bar — listed first so it wins over harami
    detect_morning_star,        # 3-bar
    detect_evening_star,        # 3-bar
    detect_three_white_soldiers,
    detect_three_black_crows,
    detect_piercing_line,
    detect_dark_cloud_cover,
    detect_hammer,
    detect_hanging_man,
    detect_shooting_star,
    detect_inverted_hammer,
    detect_marubozu,
    detect_harami,
    detect_doji,                 # listed last — weakest signal
]

# All chart pattern detectors (peak/trough based)
_CHART_PATTERN_DETECTORS = [
    detect_head_and_shoulders,
    detect_inverse_head_and_shoulders,
    detect_triple_top,
    detect_triple_bottom,
    detect_double_top,
    detect_double_bottom,
]


def detect_all_patterns(df: pd.DataFrame, timeframe: str = "daily",
                         lookback_candlesticks: int = 5,
                         peak_window: int = 5) -> dict:
    """
    Run all pattern detectors on a price DataFrame.

    Args:
        df: pandas DataFrame indexed by datetime with columns
            open/high/low/close/volume, sorted ascending.
        timeframe: "daily" | "15m" | "5m" — used to scale peak detection
            parameters (shorter timeframes need smaller peak windows).
        lookback_candlesticks: how many recent bars to scan for candlestick
            patterns (default 5 — keeps results recent).
        peak_window: window for peak/trough detection. Auto-scaled for
            intraday timeframes.

    Returns:
        {
          "timeframe":   "daily" | "15m" | "5m",
          "candlesticks": [...],   # list of recent candlestick patterns
          "chart_patterns": [...], # list of recent chart patterns
          "summary": {
              "formed_count":  N,
              "forming_count": M,
              "bullish_count": X,
              "bearish_count": Y,
          }
        }
    """
    if df is None or len(df) < 10:
        return {
            "timeframe": timeframe,
            "candlesticks": [],
            "chart_patterns": [],
            "summary": {"formed_count": 0, "forming_count": 0,
                        "bullish_count": 0, "bearish_count": 0},
        }

    # Ensure column names are lowercase and have required fields
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return {
            "timeframe": timeframe,
            "candlesticks": [],
            "chart_patterns": [],
            "summary": {"formed_count": 0, "forming_count": 0,
                        "bullish_count": 0, "bearish_count": 0},
        }
    df = df.dropna(subset=list(required))

    # Scale peak window for shorter timeframes — intraday has more noise so
    # we need a wider window to confirm a real peak
    if timeframe == "5m":
        peak_window = max(peak_window, 8)
        min_prom = 1.0
    elif timeframe == "15m":
        peak_window = max(peak_window, 6)
        min_prom = 1.2
    else:
        min_prom = 1.5

    # --- Candlestick patterns (last N bars) ---
    candle_results = []
    start = max(0, len(df) - lookback_candlesticks)
    seen_at_idx = set()  # one candlestick pattern per bar (the strongest)
    for i in range(start, len(df)):
        for detector in _CANDLESTICK_DETECTORS:
            if i in seen_at_idx:
                continue
            try:
                result = detector(df, i)
                if result:
                    candle_results.append(result)
                    seen_at_idx.add(i)
                    break  # First detector wins for this bar
            except Exception:
                continue
    # Sort by recency (smallest bars_ago first)
    candle_results.sort(key=lambda x: x["bars_ago"])

    # --- Chart patterns ---
    chart_results = []
    try:
        peaks, troughs = _find_peaks_and_troughs(df, window=peak_window,
                                                  min_prominence_pct=min_prom)
        for detector in _CHART_PATTERN_DETECTORS:
            try:
                result = detector(df, peaks, troughs)
                if result:
                    chart_results.append(result)
            except Exception:
                continue
    except Exception:
        pass
    chart_results.sort(key=lambda x: x["bars_ago"])

    # --- Summary stats ---
    all_results = candle_results + chart_results
    summary = {
        "formed_count":  sum(1 for r in all_results if r["status"] == "formed"),
        "forming_count": sum(1 for r in all_results if r["status"] == "forming"),
        "bullish_count": sum(1 for r in all_results if r["type"] == "bullish"),
        "bearish_count": sum(1 for r in all_results if r["type"] == "bearish"),
    }

    return {
        "timeframe":      timeframe,
        "candlesticks":   candle_results,
        "chart_patterns": chart_results,
        "summary":        summary,
    }
