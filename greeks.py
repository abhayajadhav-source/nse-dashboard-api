"""
Black-Scholes Greeks + Implied Volatility solver for NSE options.

Pure math library, no external dependencies beyond Python's `math` module.
Used to compute Greeks (delta/gamma/theta/vega) and reverse-solve IV when
Upstox doesn't return them in the option chain.

Conventions:
  - Risk-free rate: 6.5% annualized (10-year Indian G-sec proxy as of 2026)
  - Days to expiry expressed as integer days; converted to years internally
  - IV input/output: decimal form (0.25 = 25% annualized volatility)
  - Theta returned in "per-calendar-day" form (already divided by 365)
  - Vega returned per 1% absolute IV change (for trader-friendly scale)
  - All functions tolerate edge cases (zero DTE, deep ITM/OTM) and return
    sensible defaults rather than raising.

Reference: Hull, "Options, Futures, and Other Derivatives", chapter on Greeks.
"""

from __future__ import annotations

import math
from typing import Optional

# Default risk-free rate for Indian markets
DEFAULT_RISK_FREE_RATE = 0.065   # 6.5% annualized
# Year basis for converting DTE → years
TRADING_YEAR_DAYS = 365.0


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(spot: float, strike: float, T: float, iv: float,
           r: float) -> tuple[float, float]:
    """
    Black-Scholes d1 and d2 terms.
    T is time-to-expiry in years.
    iv is annualized volatility as a decimal.
    """
    # Guard: extremely small T or IV produces division-by-zero; bump them
    T_safe  = max(T, 1e-6)
    iv_safe = max(iv, 1e-6)
    sigma_sqrt_T = iv_safe * math.sqrt(T_safe)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv_safe * iv_safe) * T_safe) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


def black_scholes_price(spot: float, strike: float, days_to_expiry: int,
                        iv: float, opt_type: str,
                        risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """
    Theoretical Black-Scholes price for a European call or put.

    Args:
        spot: current underlying price
        strike: option strike
        days_to_expiry: integer days remaining
        iv: implied volatility as decimal (0.25 = 25%)
        opt_type: "call" or "put"
        risk_free_rate: annualized risk-free rate

    Returns:
        Theoretical premium (always >= 0).
    """
    if spot <= 0 or strike <= 0 or days_to_expiry <= 0 or iv <= 0:
        # Degenerate input — fall back to intrinsic value
        if opt_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    T = days_to_expiry / TRADING_YEAR_DAYS
    d1, d2 = _d1_d2(spot, strike, T, iv, risk_free_rate)
    discount = math.exp(-risk_free_rate * T)

    if opt_type == "call":
        return spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    else:  # put
        return strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def black_scholes_greeks(spot: float, strike: float, days_to_expiry: int,
                         iv: float, opt_type: str,
                         risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> dict:
    """
    Compute the four key Greeks (delta, gamma, theta, vega) for a European
    call or put under Black-Scholes assumptions.

    Returns a dict with keys "delta", "gamma", "theta", "vega".
    On any computation failure (degenerate inputs), returns zeros.

    Conventions for trader-friendly scale:
      - delta in [0, 1] for calls, [-1, 0] for puts
      - gamma per unit move in spot
      - theta PER CALENDAR DAY (negative for long options, positive for short)
      - vega PER 1% absolute change in IV (so vega=4.5 means premium gains
        ₹4.5 if IV moves from 25% to 26%)
    """
    out = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    if spot <= 0 or strike <= 0 or days_to_expiry <= 0 or iv <= 0:
        return out

    try:
        T = days_to_expiry / TRADING_YEAR_DAYS
        d1, d2 = _d1_d2(spot, strike, T, iv, risk_free_rate)
        pdf_d1 = _norm_pdf(d1)
        sqrt_T = math.sqrt(T)
        discount = math.exp(-risk_free_rate * T)

        # Gamma and Vega are the same for calls and puts
        gamma = pdf_d1 / (spot * iv * sqrt_T)
        # Vega in raw BS form is per 1.0 (100%) IV change; scale to per 1% so
        # trader sees "₹X per 1% IV change" which is the intuitive scale
        vega = spot * pdf_d1 * sqrt_T / 100.0

        if opt_type == "call":
            delta = _norm_cdf(d1)
            # Theta (annualized) — convert to per-calendar-day
            theta_annual = (
                -spot * pdf_d1 * iv / (2.0 * sqrt_T)
                - risk_free_rate * strike * discount * _norm_cdf(d2)
            )
        else:  # put
            delta = _norm_cdf(d1) - 1.0   # equivalent to -N(-d1)
            theta_annual = (
                -spot * pdf_d1 * iv / (2.0 * sqrt_T)
                + risk_free_rate * strike * discount * _norm_cdf(-d2)
            )

        theta_per_day = theta_annual / TRADING_YEAR_DAYS

        out["delta"] = round(delta, 4)
        out["gamma"] = round(gamma, 6)
        out["theta"] = round(theta_per_day, 4)
        out["vega"]  = round(vega, 4)
    except (ValueError, ZeroDivisionError, OverflowError):
        # Numerical instability — return zeros rather than crash
        pass

    return out


def implied_volatility(spot: float, strike: float, premium: float,
                       days_to_expiry: int, opt_type: str,
                       risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                       max_iterations: int = 100,
                       tolerance: float = 1e-4) -> Optional[float]:
    """
    Reverse-solve implied volatility from observed option premium using
    Newton-Raphson iteration.

    Returns:
        IV as decimal (0.25 = 25%), or None if solver fails to converge or
        inputs are degenerate.

    Notes:
      - Bounded to [0.5%, 500%] IV — anything outside is signal of bad data
        (e.g., LTP of a zero-OI option that's stale)
      - Initial guess: 30% (typical for Indian single-stock IVs)
      - For deep ITM/OTM options, vega → 0 and solver becomes unstable;
        we detect this and return None.
    """
    if spot <= 0 or strike <= 0 or premium <= 0 or days_to_expiry <= 0:
        return None

    # Reject premiums below intrinsic value (arbitrage violation; likely
    # stale or wrong data)
    if opt_type == "call":
        intrinsic = max(spot - strike, 0.0)
    else:
        intrinsic = max(strike - spot, 0.0)
    if premium < intrinsic - 0.01:
        return None

    T = days_to_expiry / TRADING_YEAR_DAYS

    # Newton-Raphson — iterate IV until BS price matches observed premium
    iv = 0.30  # starting guess: 30% — typical Indian single-stock IV
    for _ in range(max_iterations):
        try:
            bs_price = black_scholes_price(spot, strike, days_to_expiry, iv,
                                            opt_type, risk_free_rate)
            # Vega in raw form (per 1.0 IV change) — needed for Newton step
            d1, _ = _d1_d2(spot, strike, T, iv, risk_free_rate)
            vega_raw = spot * _norm_pdf(d1) * math.sqrt(T)

            if vega_raw < 1e-8:
                # Numerically degenerate — bail
                return None

            error = bs_price - premium
            if abs(error) < tolerance:
                # Sanity bound: reject implausible IVs
                if iv < 0.005 or iv > 5.0:
                    return None
                return round(iv, 4)

            iv -= error / vega_raw

            # Keep IV in plausible range during iteration (prevents
            # divergence into negative or absurd values)
            if iv < 0.005:
                iv = 0.005
            elif iv > 5.0:
                iv = 5.0
        except (ValueError, ZeroDivisionError, OverflowError):
            return None

    # Didn't converge in max_iterations
    return None


def compute_atm_iv(strike_summary: list, underlying_price: float,
                   days_to_expiry: int) -> Optional[float]:
    """
    Helper for the analyzer: compute ATM IV as the average of the nearest
    call and put IVs (using their existing iv field if present, else
    reverse-solving from LTP).

    Args:
        strike_summary: list of dicts with at least `strike`, `call_ltp`,
                        `put_ltp`, and optionally `call_iv`, `put_iv`
        underlying_price: current spot
        days_to_expiry: integer days

    Returns:
        ATM IV as decimal, or None if it can't be computed.
    """
    if not strike_summary or underlying_price <= 0 or days_to_expiry <= 0:
        return None

    # Find the ATM strike (closest to spot)
    atm = min(strike_summary, key=lambda s: abs(s["strike"] - underlying_price))

    call_iv = atm.get("call_iv")
    put_iv  = atm.get("put_iv")

    # If IV wasn't provided in the source data, reverse-solve from LTP
    if call_iv is None and atm.get("call_ltp", 0) > 0:
        call_iv = implied_volatility(
            underlying_price, atm["strike"], atm["call_ltp"],
            days_to_expiry, "call",
        )
    if put_iv is None and atm.get("put_ltp", 0) > 0:
        put_iv = implied_volatility(
            underlying_price, atm["strike"], atm["put_ltp"],
            days_to_expiry, "put",
        )

    # Average call + put IV (if both available); else use whichever is set
    if call_iv is not None and put_iv is not None:
        return round((call_iv + put_iv) / 2.0, 4)
    if call_iv is not None:
        return round(call_iv, 4)
    if put_iv is not None:
        return round(put_iv, 4)
    return None
