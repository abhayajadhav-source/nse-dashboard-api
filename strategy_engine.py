"""
Options Strategy Engine

Given a stock's price, IV regime, option chain, and outlook, builds concrete
strategy setups with strikes, costs, max P&L, and breakevens.

A "strategy" here is a callable that takes a `StrategyContext` (price, option
chain, outlook flags) and returns a `StrategyResult` (legs, cost, max P/L,
breakevens, fit score) — or None if the strategy doesn't make sense for the
current market context.

Each strategy is independently scored on its fit to the outlook so the AI
gets a ranked menu to pick from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class OptionStrike:
    """Single strike's market data — both call and put side."""
    strike: float
    call_ltp: float
    call_oi: int
    put_ltp:  float
    put_oi:   int
    # Distance helpers (computed once)
    distance_from_spot: float = 0.0       # +ve = OTM call / ITM put
    distance_pct:       float = 0.0       # same as % of spot


@dataclass
class StrategyContext:
    """
    Everything the strategy builders need to know about the market.
    """
    symbol: str
    spot_price: float
    lot_size:   int                        # NSE F&O lot size for this symbol
    strikes:    List[OptionStrike]         # Sorted ascending

    # Outlook flags (computed from technicals + analyst + options)
    direction:        str   # 'bullish', 'bearish', 'neutral'
    conviction:       str   # 'strong', 'moderate', 'weak'
    iv_regime:        str   # 'high', 'normal', 'low'  (proxy from PCR + composite)

    # Targets — used to size profit zones
    target_price:     float                # Analyst target or technical target
    stop_price:       float                # Technical stop (support/resistance)

    # ATR for sizing strikes by 1-σ moves
    atr_14:           float

    # Days to expiry of the chain we're analysing
    days_to_expiry:   int


@dataclass
class StrategyLeg:
    """One leg of an options strategy."""
    action:   str          # 'BUY' or 'SELL'
    quantity: int          # number of lots (not contracts)
    instrument: str        # 'CALL', 'PUT', or 'STOCK'
    strike:   Optional[float]  # None for STOCK legs
    premium:  float        # Last traded price (per share, not per lot)


@dataclass
class StrategyResult:
    """Full breakdown of a single strategy setup."""
    name:        str                    # e.g. "Bull Call Spread"
    category:    str                    # 'directional' | 'spread' | 'volatility' | ...
    legs:        List[StrategyLeg]

    # Cost & P/L (per lot, in ₹)
    net_debit:   float                  # +ve = debit (you pay), -ve = credit (you receive)
    max_profit:  Optional[float]        # None = unlimited
    max_loss:    Optional[float]        # None = unlimited
    breakevens:  List[float]            # one or more breakeven spot prices

    # Sizing
    lot_size:    int
    capital_required: float             # premium debit + margin estimate

    # Outlook fit (0-100). Higher = better fit for current market view.
    fit_score:   int
    fit_reason:  str                    # Short text justifying score

    # Risk profile (informational, for UI badges)
    risk_profile: str                   # 'defined' | 'undefined'
    direction_bias: str                 # 'bullish' | 'bearish' | 'neutral' | 'volatility'

    # Probability of profit estimate (rough, at expiry)
    prob_profit: Optional[float] = None  # 0-100, None if not computable

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Helpers for finding strikes
# ---------------------------------------------------------------------------
def _find_atm(strikes: List[OptionStrike], spot: float) -> Optional[OptionStrike]:
    """Strike closest to spot."""
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s.strike - spot))


def _find_strike_at_distance(strikes: List[OptionStrike], spot: float,
                              target_distance_pct: float) -> Optional[OptionStrike]:
    """
    Find the strike closest to (spot + spot*target_distance_pct/100).

    target_distance_pct can be negative (for OTM puts / ITM calls).
    """
    if not strikes:
        return None
    target = spot * (1 + target_distance_pct / 100)
    return min(strikes, key=lambda s: abs(s.strike - target))


def _find_strike_near(strikes: List[OptionStrike], target_price: float) -> Optional[OptionStrike]:
    """Strike closest to a specific price."""
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s.strike - target_price))


def _otm_call_strikes(strikes: List[OptionStrike], spot: float) -> List[OptionStrike]:
    """Strikes above spot (where calls are OTM)."""
    return [s for s in strikes if s.strike > spot]


def _otm_put_strikes(strikes: List[OptionStrike], spot: float) -> List[OptionStrike]:
    """Strikes below spot (where puts are OTM)."""
    return [s for s in strikes if s.strike < spot]


# ---------------------------------------------------------------------------
# Strategy Builders
# Each takes a StrategyContext and returns StrategyResult or None.
# All P&L values are PER LOT (premium × lot_size already applied).
# ---------------------------------------------------------------------------

def build_long_call(ctx: StrategyContext) -> Optional[StrategyResult]:
    """
    Long Call: Buy ATM or slightly OTM call.
    Best for: strong bullish view, moderate IV.
    Risk: limited to premium paid. Reward: unlimited.
    """
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    if not atm or atm.call_ltp <= 0:
        return None

    premium = atm.call_ltp
    cost = premium * ctx.lot_size

    # Fit score: high if strongly bullish + low/normal IV
    fit = 0
    reasons = []
    if ctx.direction == 'bullish':
        fit += 40
        reasons.append("Strong directional bullish view")
    elif ctx.direction == 'neutral':
        fit += 15
    if ctx.conviction == 'strong': fit += 25
    elif ctx.conviction == 'moderate': fit += 15
    if ctx.iv_regime in ('low', 'normal'): fit += 20  # Buying options favors low IV
    elif ctx.iv_regime == 'high': fit -= 10
    if ctx.days_to_expiry >= 10: fit += 10  # Need time

    return StrategyResult(
        name="Long Call",
        category="directional",
        legs=[StrategyLeg('BUY', 1, 'CALL', atm.strike, premium)],
        net_debit=cost,
        max_profit=None,                                  # Unlimited
        max_loss=cost,                                    # Premium paid
        breakevens=[atm.strike + premium],
        lot_size=ctx.lot_size,
        capital_required=cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Bullish positioning",
        risk_profile='defined',
        direction_bias='bullish',
    )


def build_long_put(ctx: StrategyContext) -> Optional[StrategyResult]:
    """
    Long Put: Buy ATM or slightly OTM put.
    Best for: strong bearish view, moderate IV.
    Risk: limited to premium paid. Reward: large (down to 0).
    """
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    if not atm or atm.put_ltp <= 0:
        return None

    premium = atm.put_ltp
    cost = premium * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bearish':
        fit += 40
        reasons.append("Strong directional bearish view")
    elif ctx.direction == 'neutral':
        fit += 15
    if ctx.conviction == 'strong': fit += 25
    elif ctx.conviction == 'moderate': fit += 15
    if ctx.iv_regime in ('low', 'normal'): fit += 20
    elif ctx.iv_regime == 'high': fit -= 10
    if ctx.days_to_expiry >= 10: fit += 10

    return StrategyResult(
        name="Long Put",
        category="directional",
        legs=[StrategyLeg('BUY', 1, 'PUT', atm.strike, premium)],
        net_debit=cost,
        max_profit=(atm.strike - premium) * ctx.lot_size,  # If spot goes to 0
        max_loss=cost,
        breakevens=[atm.strike - premium],
        lot_size=ctx.lot_size,
        capital_required=cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Bearish positioning",
        risk_profile='defined',
        direction_bias='bearish',
    )


def build_covered_call(ctx: StrategyContext) -> Optional[StrategyResult]:
    """
    Covered Call: Long stock + Short OTM call.
    Best for: own the stock, mildly bullish to neutral, want extra income.
    """
    otm = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=3)
    if not otm or otm.call_ltp <= 0:
        return None

    premium = otm.call_ltp
    credit = premium * ctx.lot_size
    stock_cost = ctx.spot_price * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction in ('neutral', 'bullish'):
        fit += 30
        reasons.append("Generates income on existing/new long stock")
    if ctx.iv_regime == 'high':
        fit += 25
        reasons.append("High IV makes call premium attractive")
    elif ctx.iv_regime == 'normal':
        fit += 10
    if ctx.conviction == 'moderate':
        fit += 15
        reasons.append("Caps upside — better for moderate conviction")
    if ctx.days_to_expiry <= 30: fit += 15  # Short-dated is better for theta

    max_profit = (otm.strike - ctx.spot_price + premium) * ctx.lot_size
    return StrategyResult(
        name="Covered Call",
        category="income",
        legs=[
            StrategyLeg('BUY',  1, 'STOCK', None,       ctx.spot_price),
            StrategyLeg('SELL', 1, 'CALL',  otm.strike, premium),
        ],
        net_debit=stock_cost - credit,
        max_profit=max_profit,
        max_loss=(ctx.spot_price - premium) * ctx.lot_size,  # If stock → 0
        breakevens=[ctx.spot_price - premium],
        lot_size=ctx.lot_size,
        capital_required=stock_cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Income on long stock",
        risk_profile='defined',
        direction_bias='neutral',
    )


def build_covered_put(ctx: StrategyContext) -> Optional[StrategyResult]:
    """
    Covered Put: Short stock + Short OTM put.
    Best for: short stock position, mildly bearish to neutral.
    Note: Requires margin to short stock — not common in Indian retail context.
    """
    otm = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-3)
    if not otm or otm.put_ltp <= 0:
        return None

    premium = otm.put_ltp
    credit = premium * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction in ('neutral', 'bearish'):
        fit += 25
        reasons.append("Generates income on short stock")
    if ctx.iv_regime == 'high':
        fit += 20
    # Penalize because shorting stock in India is futures-only / complex
    fit -= 10
    reasons.append("Note: requires futures short or stock loan in Indian market")

    return StrategyResult(
        name="Covered Put",
        category="income",
        legs=[
            StrategyLeg('SELL', 1, 'STOCK', None,       ctx.spot_price),
            StrategyLeg('SELL', 1, 'PUT',   otm.strike, premium),
        ],
        net_debit=-credit,  # Net credit
        max_profit=(ctx.spot_price - otm.strike + premium) * ctx.lot_size,
        max_loss=None,  # Stock can rise unlimited
        breakevens=[ctx.spot_price + premium],
        lot_size=ctx.lot_size,
        capital_required=ctx.spot_price * ctx.lot_size,  # Approximate margin
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons),
        risk_profile='undefined',
        direction_bias='bearish',
    )


def build_bull_call_spread(ctx: StrategyContext) -> Optional[StrategyResult]:
    """
    Bull Call Spread: Buy ATM call, Sell OTM call.
    Best for: moderately bullish view, want defined risk, cheaper than long call.
    """
    long_strike = _find_atm(ctx.strikes, ctx.spot_price)
    short_strike = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=5)
    if not long_strike or not short_strike or long_strike.strike >= short_strike.strike:
        return None
    if long_strike.call_ltp <= 0 or short_strike.call_ltp <= 0:
        return None

    long_p = long_strike.call_ltp
    short_p = short_strike.call_ltp
    net_debit_per_share = long_p - short_p
    if net_debit_per_share <= 0:
        return None  # Sanity — should always be positive

    net_debit = net_debit_per_share * ctx.lot_size
    width = short_strike.strike - long_strike.strike
    max_profit = (width - net_debit_per_share) * ctx.lot_size
    max_loss = net_debit

    fit = 0
    reasons = []
    if ctx.direction == 'bullish':
        fit += 35
        reasons.append("Bullish bias with defined risk")
    if ctx.conviction in ('moderate', 'strong'): fit += 20
    if ctx.iv_regime == 'high':
        fit += 20
        reasons.append("High IV — selling part of premium offsets")
    elif ctx.iv_regime == 'normal':
        fit += 10
    if ctx.days_to_expiry >= 7: fit += 10

    return StrategyResult(
        name="Bull Call Spread",
        category="spread",
        legs=[
            StrategyLeg('BUY',  1, 'CALL', long_strike.strike,  long_p),
            StrategyLeg('SELL', 1, 'CALL', short_strike.strike, short_p),
        ],
        net_debit=net_debit,
        max_profit=max_profit,
        max_loss=max_loss,
        breakevens=[long_strike.strike + net_debit_per_share],
        lot_size=ctx.lot_size,
        capital_required=net_debit,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Bullish defined-risk",
        risk_profile='defined',
        direction_bias='bullish',
    )


def build_bear_put_spread(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Bear Put Spread: Buy ATM put, Sell OTM put."""
    long_strike = _find_atm(ctx.strikes, ctx.spot_price)
    short_strike = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-5)
    if not long_strike or not short_strike or long_strike.strike <= short_strike.strike:
        return None
    if long_strike.put_ltp <= 0 or short_strike.put_ltp <= 0:
        return None

    long_p = long_strike.put_ltp
    short_p = short_strike.put_ltp
    net_debit_per_share = long_p - short_p
    if net_debit_per_share <= 0:
        return None

    net_debit = net_debit_per_share * ctx.lot_size
    width = long_strike.strike - short_strike.strike
    max_profit = (width - net_debit_per_share) * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bearish':
        fit += 35
        reasons.append("Bearish bias with defined risk")
    if ctx.conviction in ('moderate', 'strong'): fit += 20
    if ctx.iv_regime == 'high':
        fit += 20
        reasons.append("High IV — short leg offsets long premium")
    elif ctx.iv_regime == 'normal':
        fit += 10
    if ctx.days_to_expiry >= 7: fit += 10

    return StrategyResult(
        name="Bear Put Spread",
        category="spread",
        legs=[
            StrategyLeg('BUY',  1, 'PUT', long_strike.strike,  long_p),
            StrategyLeg('SELL', 1, 'PUT', short_strike.strike, short_p),
        ],
        net_debit=net_debit,
        max_profit=max_profit,
        max_loss=net_debit,
        breakevens=[long_strike.strike - net_debit_per_share],
        lot_size=ctx.lot_size,
        capital_required=net_debit,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Bearish defined-risk",
        risk_profile='defined',
        direction_bias='bearish',
    )


def build_bull_put_spread(ctx: StrategyContext) -> Optional[StrategyResult]:
    """
    Bull Put Spread (credit spread): Sell OTM put, Buy further OTM put.
    Best for: moderately bullish, want to collect premium.
    """
    short_strike = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-3)
    long_strike  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-8)
    if not short_strike or not long_strike or short_strike.strike <= long_strike.strike:
        return None
    if short_strike.put_ltp <= 0 or long_strike.put_ltp <= 0:
        return None

    credit_per_share = short_strike.put_ltp - long_strike.put_ltp
    if credit_per_share <= 0:
        return None

    credit = credit_per_share * ctx.lot_size
    width = short_strike.strike - long_strike.strike
    max_loss = (width - credit_per_share) * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bullish':
        fit += 30
        reasons.append("Income on bullish view")
    elif ctx.direction == 'neutral':
        fit += 20
    if ctx.iv_regime == 'high':
        fit += 25
        reasons.append("High IV — premium worth selling")
    elif ctx.iv_regime == 'normal':
        fit += 10
    if ctx.conviction == 'strong':
        fit += 15
    if ctx.days_to_expiry <= 30: fit += 15  # Theta works for us

    return StrategyResult(
        name="Bull Put Spread",
        category="credit_spread",
        legs=[
            StrategyLeg('SELL', 1, 'PUT', short_strike.strike, short_strike.put_ltp),
            StrategyLeg('BUY',  1, 'PUT', long_strike.strike,  long_strike.put_ltp),
        ],
        net_debit=-credit,  # Credit
        max_profit=credit,
        max_loss=max_loss,
        breakevens=[short_strike.strike - credit_per_share],
        lot_size=ctx.lot_size,
        capital_required=max_loss,  # Margin = max loss
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Bullish credit",
        risk_profile='defined',
        direction_bias='bullish',
    )


def build_bear_call_spread(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Bear Call Spread (credit): Sell OTM call, Buy further OTM call."""
    short_strike = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=3)
    long_strike  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=8)
    if not short_strike or not long_strike or short_strike.strike >= long_strike.strike:
        return None
    if short_strike.call_ltp <= 0 or long_strike.call_ltp <= 0:
        return None

    credit_per_share = short_strike.call_ltp - long_strike.call_ltp
    if credit_per_share <= 0:
        return None

    credit = credit_per_share * ctx.lot_size
    width = long_strike.strike - short_strike.strike
    max_loss = (width - credit_per_share) * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bearish':
        fit += 30
        reasons.append("Income on bearish view")
    elif ctx.direction == 'neutral':
        fit += 20
    if ctx.iv_regime == 'high':
        fit += 25
        reasons.append("High IV makes credit attractive")
    elif ctx.iv_regime == 'normal':
        fit += 10
    if ctx.conviction == 'strong':
        fit += 15
    if ctx.days_to_expiry <= 30: fit += 15

    return StrategyResult(
        name="Bear Call Spread",
        category="credit_spread",
        legs=[
            StrategyLeg('SELL', 1, 'CALL', short_strike.strike, short_strike.call_ltp),
            StrategyLeg('BUY',  1, 'CALL', long_strike.strike,  long_strike.call_ltp),
        ],
        net_debit=-credit,
        max_profit=credit,
        max_loss=max_loss,
        breakevens=[short_strike.strike + credit_per_share],
        lot_size=ctx.lot_size,
        capital_required=max_loss,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Bearish credit",
        risk_profile='defined',
        direction_bias='bearish',
    )


def build_protective_put(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Long stock + Long OTM put (portfolio insurance)."""
    put_strike = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-5)
    if not put_strike or put_strike.put_ltp <= 0:
        return None

    premium = put_strike.put_ltp
    cost = (ctx.spot_price + premium) * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bullish':
        fit += 25
        reasons.append("Hedge for existing/new long position")
    if ctx.iv_regime == 'low':
        fit += 25
        reasons.append("Low IV — cheap insurance")
    elif ctx.iv_regime == 'normal':
        fit += 10
    if ctx.conviction == 'weak':
        fit += 15
        reasons.append("Weak conviction — insurance valuable")

    return StrategyResult(
        name="Protective Put",
        category="hedge",
        legs=[
            StrategyLeg('BUY', 1, 'STOCK', None,             ctx.spot_price),
            StrategyLeg('BUY', 1, 'PUT',   put_strike.strike, premium),
        ],
        net_debit=cost,
        max_profit=None,  # Unlimited upside from stock
        max_loss=(ctx.spot_price - put_strike.strike + premium) * ctx.lot_size,
        breakevens=[ctx.spot_price + premium],
        lot_size=ctx.lot_size,
        capital_required=cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Insurance on long",
        risk_profile='defined',
        direction_bias='bullish',
    )


def build_collar(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Long stock + Long OTM put + Short OTM call (zero-cost protection)."""
    put_strike  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-5)
    call_strike = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=5)
    if not put_strike or not call_strike:
        return None
    if put_strike.put_ltp <= 0 or call_strike.call_ltp <= 0:
        return None

    net_option_cost_per_share = put_strike.put_ltp - call_strike.call_ltp
    stock_cost = ctx.spot_price * ctx.lot_size
    net_cost = stock_cost + net_option_cost_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction in ('neutral', 'bullish'):
        fit += 25
        reasons.append("Caps both upside and downside")
    if ctx.iv_regime == 'high':
        fit += 25
        reasons.append("Selling call funds the put")
    if ctx.conviction == 'weak': fit += 20

    max_profit = (call_strike.strike - ctx.spot_price - net_option_cost_per_share) * ctx.lot_size
    max_loss   = (ctx.spot_price - put_strike.strike + net_option_cost_per_share) * ctx.lot_size

    return StrategyResult(
        name="Collar Strategy",
        category="hedge",
        legs=[
            StrategyLeg('BUY',  1, 'STOCK', None,               ctx.spot_price),
            StrategyLeg('BUY',  1, 'PUT',   put_strike.strike,  put_strike.put_ltp),
            StrategyLeg('SELL', 1, 'CALL',  call_strike.strike, call_strike.call_ltp),
        ],
        net_debit=net_cost,
        max_profit=max_profit,
        max_loss=max_loss,
        breakevens=[ctx.spot_price + net_option_cost_per_share],
        lot_size=ctx.lot_size,
        capital_required=stock_cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Bracketed risk",
        risk_profile='defined',
        direction_bias='neutral',
    )


def build_long_straddle(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Buy ATM call + Buy ATM put. Profits from large move either way."""
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    if not atm or atm.call_ltp <= 0 or atm.put_ltp <= 0:
        return None

    total_premium_per_share = atm.call_ltp + atm.put_ltp
    cost = total_premium_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.iv_regime == 'low':
        fit += 35
        reasons.append("Low IV — cheap premium, room for IV expansion")
    elif ctx.iv_regime == 'normal':
        fit += 15
    elif ctx.iv_regime == 'high':
        fit -= 20  # Bad to buy straddle in high IV
    if ctx.direction == 'neutral':
        fit += 20
        reasons.append("Pre-event / unclear direction")
    if ctx.conviction == 'weak':
        fit += 15
    if ctx.days_to_expiry >= 14:
        fit += 10

    return StrategyResult(
        name="Long Straddle",
        category="volatility",
        legs=[
            StrategyLeg('BUY', 1, 'CALL', atm.strike, atm.call_ltp),
            StrategyLeg('BUY', 1, 'PUT',  atm.strike, atm.put_ltp),
        ],
        net_debit=cost,
        max_profit=None,  # Unlimited (call side)
        max_loss=cost,
        breakevens=[atm.strike - total_premium_per_share,
                    atm.strike + total_premium_per_share],
        lot_size=ctx.lot_size,
        capital_required=cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Volatility expansion bet",
        risk_profile='defined',
        direction_bias='volatility',
    )


def build_short_straddle(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Sell ATM call + Sell ATM put. Profits from low realized vol. Risky — undefined."""
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    if not atm or atm.call_ltp <= 0 or atm.put_ltp <= 0:
        return None

    total_premium_per_share = atm.call_ltp + atm.put_ltp
    credit = total_premium_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.iv_regime == 'high':
        fit += 35
        reasons.append("High IV — premium worth collecting")
    elif ctx.iv_regime == 'normal':
        fit += 10
    elif ctx.iv_regime == 'low':
        fit -= 25
    if ctx.direction == 'neutral':
        fit += 25
        reasons.append("Sideways outlook — collects theta")
    if ctx.days_to_expiry <= 14:
        fit += 15
    fit -= 15  # Always penalize naked short — undefined risk

    return StrategyResult(
        name="Short Straddle",
        category="volatility",
        legs=[
            StrategyLeg('SELL', 1, 'CALL', atm.strike, atm.call_ltp),
            StrategyLeg('SELL', 1, 'PUT',  atm.strike, atm.put_ltp),
        ],
        net_debit=-credit,
        max_profit=credit,
        max_loss=None,  # Unlimited
        breakevens=[atm.strike - total_premium_per_share,
                    atm.strike + total_premium_per_share],
        lot_size=ctx.lot_size,
        capital_required=credit * 3,  # Rough margin estimate (3x credit)
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Theta + IV crush",
        risk_profile='undefined',
        direction_bias='neutral',
    )


def build_long_strangle(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Buy OTM call + Buy OTM put. Cheaper than straddle, needs bigger move."""
    call_strike = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=3)
    put_strike  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-3)
    if not call_strike or not put_strike:
        return None
    if call_strike.call_ltp <= 0 or put_strike.put_ltp <= 0:
        return None

    total_per_share = call_strike.call_ltp + put_strike.put_ltp
    cost = total_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.iv_regime == 'low':
        fit += 30
        reasons.append("Low IV — cheap volatility exposure")
    elif ctx.iv_regime == 'normal':
        fit += 10
    elif ctx.iv_regime == 'high':
        fit -= 15
    if ctx.direction == 'neutral':
        fit += 20
        reasons.append("Big move expected, direction unclear")
    if ctx.days_to_expiry >= 14: fit += 10

    return StrategyResult(
        name="Long Strangle",
        category="volatility",
        legs=[
            StrategyLeg('BUY', 1, 'CALL', call_strike.strike, call_strike.call_ltp),
            StrategyLeg('BUY', 1, 'PUT',  put_strike.strike,  put_strike.put_ltp),
        ],
        net_debit=cost,
        max_profit=None,
        max_loss=cost,
        breakevens=[put_strike.strike - total_per_share,
                    call_strike.strike + total_per_share],
        lot_size=ctx.lot_size,
        capital_required=cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Cheaper vol bet",
        risk_profile='defined',
        direction_bias='volatility',
    )


def build_short_strangle(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Sell OTM call + Sell OTM put. Common income strategy."""
    call_strike = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=3)
    put_strike  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-3)
    if not call_strike or not put_strike:
        return None
    if call_strike.call_ltp <= 0 or put_strike.put_ltp <= 0:
        return None

    total_per_share = call_strike.call_ltp + put_strike.put_ltp
    credit = total_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.iv_regime == 'high':
        fit += 35
        reasons.append("High IV — premium attractive")
    elif ctx.iv_regime == 'normal':
        fit += 15
    elif ctx.iv_regime == 'low':
        fit -= 15
    if ctx.direction == 'neutral':
        fit += 25
        reasons.append("Range-bound expectation")
    if ctx.days_to_expiry <= 21: fit += 15
    fit -= 10  # Undefined risk penalty (less than straddle)

    return StrategyResult(
        name="Short Strangle",
        category="volatility",
        legs=[
            StrategyLeg('SELL', 1, 'CALL', call_strike.strike, call_strike.call_ltp),
            StrategyLeg('SELL', 1, 'PUT',  put_strike.strike,  put_strike.put_ltp),
        ],
        net_debit=-credit,
        max_profit=credit,
        max_loss=None,
        breakevens=[put_strike.strike - total_per_share,
                    call_strike.strike + total_per_share],
        lot_size=ctx.lot_size,
        capital_required=credit * 3,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Range income",
        risk_profile='undefined',
        direction_bias='neutral',
    )


def build_iron_condor(ctx: StrategyContext) -> Optional[StrategyResult]:
    """
    Iron Condor: Bull put spread + Bear call spread.
    Sells inner strikes, buys outer wings. Defined risk version of short strangle.
    Most popular range-bound strategy.
    """
    # Short put 3% OTM, long put 8% OTM
    short_put  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-3)
    long_put   = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-8)
    short_call = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=3)
    long_call  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=8)
    if not all([short_put, long_put, short_call, long_call]): return None
    if not (long_put.strike < short_put.strike < short_call.strike < long_call.strike):
        return None
    # All premiums must be positive
    if min(short_put.put_ltp, long_put.put_ltp, short_call.call_ltp, long_call.call_ltp) <= 0:
        return None

    credit_per_share = (short_put.put_ltp - long_put.put_ltp) + (short_call.call_ltp - long_call.call_ltp)
    if credit_per_share <= 0:
        return None

    credit = credit_per_share * ctx.lot_size
    width_put  = short_put.strike - long_put.strike
    width_call = long_call.strike - short_call.strike
    max_loss = (max(width_put, width_call) - credit_per_share) * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'neutral':
        fit += 35
        reasons.append("Neutral range-bound outlook")
    if ctx.iv_regime == 'high':
        fit += 30
        reasons.append("High IV — premium worth collecting")
    elif ctx.iv_regime == 'normal':
        fit += 15
    if ctx.days_to_expiry <= 30 and ctx.days_to_expiry >= 7: fit += 15

    return StrategyResult(
        name="Iron Condor",
        category="condor",
        legs=[
            StrategyLeg('BUY',  1, 'PUT',  long_put.strike,   long_put.put_ltp),
            StrategyLeg('SELL', 1, 'PUT',  short_put.strike,  short_put.put_ltp),
            StrategyLeg('SELL', 1, 'CALL', short_call.strike, short_call.call_ltp),
            StrategyLeg('BUY',  1, 'CALL', long_call.strike,  long_call.call_ltp),
        ],
        net_debit=-credit,
        max_profit=credit,
        max_loss=max_loss,
        breakevens=[short_put.strike  - credit_per_share,
                    short_call.strike + credit_per_share],
        lot_size=ctx.lot_size,
        capital_required=max_loss,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Neutral defined-risk",
        risk_profile='defined',
        direction_bias='neutral',
    )


def build_iron_butterfly(ctx: StrategyContext) -> Optional[StrategyResult]:
    """
    Iron Butterfly: Sell ATM straddle + Buy OTM strangle wings.
    Like iron condor but with shorts at ATM (higher credit, narrower profit zone).
    """
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    long_put  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-5)
    long_call = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=5)
    if not atm or not long_put or not long_call: return None
    if min(atm.call_ltp, atm.put_ltp, long_put.put_ltp, long_call.call_ltp) <= 0:
        return None

    credit_per_share = (atm.call_ltp + atm.put_ltp) - (long_put.put_ltp + long_call.call_ltp)
    if credit_per_share <= 0: return None

    credit = credit_per_share * ctx.lot_size
    width = max(atm.strike - long_put.strike, long_call.strike - atm.strike)
    max_loss = (width - credit_per_share) * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'neutral':
        fit += 30
        reasons.append("Tight range / pin expected")
    if ctx.iv_regime == 'high':
        fit += 30
    elif ctx.iv_regime == 'normal':
        fit += 10
    if ctx.days_to_expiry <= 14: fit += 20

    return StrategyResult(
        name="Iron Butterfly",
        category="butterfly",
        legs=[
            StrategyLeg('BUY',  1, 'PUT',  long_put.strike,  long_put.put_ltp),
            StrategyLeg('SELL', 1, 'PUT',  atm.strike,       atm.put_ltp),
            StrategyLeg('SELL', 1, 'CALL', atm.strike,       atm.call_ltp),
            StrategyLeg('BUY',  1, 'CALL', long_call.strike, long_call.call_ltp),
        ],
        net_debit=-credit,
        max_profit=credit,
        max_loss=max_loss,
        breakevens=[atm.strike - credit_per_share, atm.strike + credit_per_share],
        lot_size=ctx.lot_size,
        capital_required=max_loss,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Pin to ATM",
        risk_profile='defined',
        direction_bias='neutral',
    )


def build_butterfly_spread(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Long Call Butterfly: Buy 1 ITM, Sell 2 ATM, Buy 1 OTM (all calls)."""
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    itm = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-3)
    otm = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=3)
    if not atm or not itm or not otm: return None
    if not (itm.strike < atm.strike < otm.strike): return None
    if min(atm.call_ltp, itm.call_ltp, otm.call_ltp) <= 0: return None

    cost_per_share = itm.call_ltp - 2*atm.call_ltp + otm.call_ltp
    if cost_per_share <= 0: return None

    cost = cost_per_share * ctx.lot_size
    width = atm.strike - itm.strike
    max_profit = (width - cost_per_share) * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'neutral':
        fit += 35
        reasons.append("Pin expected near ATM")
    if ctx.iv_regime == 'low':
        fit += 20
        reasons.append("Low IV makes butterfly cheap")
    if ctx.days_to_expiry <= 21: fit += 20

    return StrategyResult(
        name="Butterfly Spread",
        category="butterfly",
        legs=[
            StrategyLeg('BUY',  1, 'CALL', itm.strike, itm.call_ltp),
            StrategyLeg('SELL', 2, 'CALL', atm.strike, atm.call_ltp),
            StrategyLeg('BUY',  1, 'CALL', otm.strike, otm.call_ltp),
        ],
        net_debit=cost,
        max_profit=max_profit,
        max_loss=cost,
        breakevens=[itm.strike + cost_per_share, otm.strike - cost_per_share],
        lot_size=ctx.lot_size,
        capital_required=cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Cheap pin bet",
        risk_profile='defined',
        direction_bias='neutral',
    )


def build_jade_lizard(ctx: StrategyContext) -> Optional[StrategyResult]:
    """
    Jade Lizard: Short OTM put + Bear call spread.
    No upside risk if credit > call spread width.
    """
    short_put  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-5)
    short_call = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=3)
    long_call  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=6)
    if not all([short_put, short_call, long_call]): return None
    if short_call.strike >= long_call.strike: return None
    if min(short_put.put_ltp, short_call.call_ltp, long_call.call_ltp) <= 0: return None

    credit_per_share = short_put.put_ltp + (short_call.call_ltp - long_call.call_ltp)
    if credit_per_share <= 0: return None

    credit = credit_per_share * ctx.lot_size
    call_width = long_call.strike - short_call.strike
    # If credit > call spread width, no upside risk; downside = strike - credit
    upside_risk = (call_width - credit_per_share) * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction in ('neutral', 'bullish'):
        fit += 30
        reasons.append("Neutral-to-bullish income")
    if ctx.iv_regime == 'high':
        fit += 30
        reasons.append("High IV makes credit attractive")
    if credit_per_share > call_width:
        fit += 15
        reasons.append("No upside risk (credit > call width)")

    return StrategyResult(
        name="Jade Lizard",
        category="hybrid",
        legs=[
            StrategyLeg('SELL', 1, 'PUT',  short_put.strike,  short_put.put_ltp),
            StrategyLeg('SELL', 1, 'CALL', short_call.strike, short_call.call_ltp),
            StrategyLeg('BUY',  1, 'CALL', long_call.strike,  long_call.call_ltp),
        ],
        net_debit=-credit,
        max_profit=credit,
        max_loss=(short_put.strike - credit_per_share) * ctx.lot_size,  # If spot → 0
        breakevens=[short_put.strike - credit_per_share],
        lot_size=ctx.lot_size,
        capital_required=max(upside_risk, (short_put.strike - credit_per_share) * ctx.lot_size * 0.2),  # Approx margin
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Asymmetric income",
        risk_profile='defined' if credit_per_share >= call_width else 'undefined',
        direction_bias='neutral',
    )


def build_reverse_iron_condor(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Reverse Iron Condor: Long inner strikes, short outer wings. Profits from big move."""
    long_put   = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-3)
    short_put  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-8)
    long_call  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=3)
    short_call = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=8)
    if not all([long_put, short_put, long_call, short_call]): return None
    if not (short_put.strike < long_put.strike < long_call.strike < short_call.strike):
        return None
    if min(long_put.put_ltp, short_put.put_ltp, long_call.call_ltp, short_call.call_ltp) <= 0:
        return None

    cost_per_share = (long_put.put_ltp - short_put.put_ltp) + (long_call.call_ltp - short_call.call_ltp)
    if cost_per_share <= 0: return None

    cost = cost_per_share * ctx.lot_size
    width_put  = long_put.strike - short_put.strike
    width_call = short_call.strike - long_call.strike
    max_profit = (min(width_put, width_call) - cost_per_share) * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.iv_regime == 'low':
        fit += 30
        reasons.append("Low IV — buy vol cheap")
    if ctx.direction == 'neutral':
        fit += 20
        reasons.append("Expected big move, direction unclear")
    if ctx.days_to_expiry >= 14: fit += 15

    return StrategyResult(
        name="Reverse Iron Condor",
        category="condor",
        legs=[
            StrategyLeg('SELL', 1, 'PUT',  short_put.strike,  short_put.put_ltp),
            StrategyLeg('BUY',  1, 'PUT',  long_put.strike,   long_put.put_ltp),
            StrategyLeg('BUY',  1, 'CALL', long_call.strike,  long_call.call_ltp),
            StrategyLeg('SELL', 1, 'CALL', short_call.strike, short_call.call_ltp),
        ],
        net_debit=cost,
        max_profit=max_profit,
        max_loss=cost,
        breakevens=[long_put.strike  - cost_per_share,
                    long_call.strike + cost_per_share],
        lot_size=ctx.lot_size,
        capital_required=cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Defined-risk volatility",
        risk_profile='defined',
        direction_bias='volatility',
    )


def build_synthetic_long(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Synthetic Long: Long ATM call + Short ATM put. Replicates long stock."""
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    if not atm or atm.call_ltp <= 0 or atm.put_ltp <= 0: return None

    net_per_share = atm.call_ltp - atm.put_ltp  # Debit if call > put (typical)
    cost = net_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bullish':
        fit += 30
        reasons.append("Synthetic long exposure")
    if ctx.conviction == 'strong': fit += 20
    fit -= 5  # Less common in retail; complicated margin

    return StrategyResult(
        name="Synthetic Long Stock",
        category="synthetic",
        legs=[
            StrategyLeg('BUY',  1, 'CALL', atm.strike, atm.call_ltp),
            StrategyLeg('SELL', 1, 'PUT',  atm.strike, atm.put_ltp),
        ],
        net_debit=cost,
        max_profit=None,
        max_loss=None,
        breakevens=[atm.strike + net_per_share],
        lot_size=ctx.lot_size,
        capital_required=atm.strike * ctx.lot_size * 0.2,  # Margin approx
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Long exposure via options",
        risk_profile='undefined',
        direction_bias='bullish',
    )


def build_synthetic_short(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Synthetic Short: Short ATM call + Long ATM put."""
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    if not atm or atm.call_ltp <= 0 or atm.put_ltp <= 0: return None

    net_per_share = atm.put_ltp - atm.call_ltp
    cost = net_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bearish':
        fit += 30
        reasons.append("Synthetic short exposure")
    if ctx.conviction == 'strong': fit += 20

    return StrategyResult(
        name="Synthetic Short Stock",
        category="synthetic",
        legs=[
            StrategyLeg('SELL', 1, 'CALL', atm.strike, atm.call_ltp),
            StrategyLeg('BUY',  1, 'PUT',  atm.strike, atm.put_ltp),
        ],
        net_debit=cost,
        max_profit=None,
        max_loss=None,
        breakevens=[atm.strike - net_per_share],
        lot_size=ctx.lot_size,
        capital_required=atm.strike * ctx.lot_size * 0.2,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Short exposure via options",
        risk_profile='undefined',
        direction_bias='bearish',
    )


def build_covered_strangle(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Long stock + Short OTM call + Short OTM put. Income with downside hedge."""
    call_strike = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=3)
    put_strike  = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=-5)
    if not call_strike or not put_strike: return None
    if call_strike.call_ltp <= 0 or put_strike.put_ltp <= 0: return None

    total_credit_per_share = call_strike.call_ltp + put_strike.put_ltp
    credit = total_credit_per_share * ctx.lot_size
    stock_cost = ctx.spot_price * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction in ('neutral', 'bullish'):
        fit += 25
        reasons.append("Aggressive income on long stock")
    if ctx.iv_regime == 'high':
        fit += 25
        reasons.append("High IV worth selling both sides")

    max_profit = (call_strike.strike - ctx.spot_price + total_credit_per_share) * ctx.lot_size
    return StrategyResult(
        name="Covered Strangle",
        category="income",
        legs=[
            StrategyLeg('BUY',  1, 'STOCK', None,               ctx.spot_price),
            StrategyLeg('SELL', 1, 'CALL',  call_strike.strike, call_strike.call_ltp),
            StrategyLeg('SELL', 1, 'PUT',   put_strike.strike,  put_strike.put_ltp),
        ],
        net_debit=stock_cost - credit,
        max_profit=max_profit,
        max_loss=(ctx.spot_price + put_strike.strike - total_credit_per_share) * ctx.lot_size,  # Both put assigned + stock drops
        breakevens=[ctx.spot_price - total_credit_per_share / 2],  # Simplified
        lot_size=ctx.lot_size,
        capital_required=stock_cost + (put_strike.strike * ctx.lot_size * 0.2),  # Stock + put margin
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Aggressive income",
        risk_profile='undefined',
        direction_bias='neutral',
    )


def build_strip(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Strip: Long 1 ATM call + Long 2 ATM puts. Bearish straddle variant."""
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    if not atm or atm.call_ltp <= 0 or atm.put_ltp <= 0: return None

    cost_per_share = atm.call_ltp + 2 * atm.put_ltp
    cost = cost_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bearish':
        fit += 30
        reasons.append("Bearish bias with vol expansion")
    if ctx.iv_regime == 'low':
        fit += 25
        reasons.append("Low IV — buy vol cheap")
    if ctx.conviction == 'weak': fit += 10  # Direction unclear, but bearish lean

    return StrategyResult(
        name="Strip Strategy",
        category="volatility",
        legs=[
            StrategyLeg('BUY', 1, 'CALL', atm.strike, atm.call_ltp),
            StrategyLeg('BUY', 2, 'PUT',  atm.strike, atm.put_ltp),
        ],
        net_debit=cost,
        max_profit=None,
        max_loss=cost,
        breakevens=[atm.strike + cost_per_share,
                    atm.strike - cost_per_share / 2],
        lot_size=ctx.lot_size,
        capital_required=cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Bearish vol",
        risk_profile='defined',
        direction_bias='bearish',
    )


def build_strap(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Strap: Long 2 ATM calls + Long 1 ATM put. Bullish straddle variant."""
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    if not atm or atm.call_ltp <= 0 or atm.put_ltp <= 0: return None

    cost_per_share = 2 * atm.call_ltp + atm.put_ltp
    cost = cost_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bullish':
        fit += 30
        reasons.append("Bullish bias with vol expansion")
    if ctx.iv_regime == 'low':
        fit += 25
    if ctx.conviction == 'weak': fit += 10

    return StrategyResult(
        name="Strap Strategy",
        category="volatility",
        legs=[
            StrategyLeg('BUY', 2, 'CALL', atm.strike, atm.call_ltp),
            StrategyLeg('BUY', 1, 'PUT',  atm.strike, atm.put_ltp),
        ],
        net_debit=cost,
        max_profit=None,
        max_loss=cost,
        breakevens=[atm.strike + cost_per_share / 2,
                    atm.strike - cost_per_share],
        lot_size=ctx.lot_size,
        capital_required=cost,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Bullish vol",
        risk_profile='defined',
        direction_bias='bullish',
    )


def build_christmas_tree(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Christmas Tree (call): Buy 1 ITM call, skip strike, Sell 3 ATM calls, Buy 2 OTM calls."""
    # Simplified version — full requires ladder of 4-5 strikes
    return None  # Placeholder — needs richer strike chain support


def build_ratio_spread(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Call Ratio Spread: Buy 1 ATM call, Sell 2 OTM calls (1x2 ratio)."""
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    otm = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=5)
    if not atm or not otm or atm.strike >= otm.strike: return None
    if atm.call_ltp <= 0 or otm.call_ltp <= 0: return None

    net_per_share = atm.call_ltp - 2 * otm.call_ltp  # Often a credit
    cost = net_per_share * ctx.lot_size
    width = otm.strike - atm.strike

    fit = 0
    reasons = []
    if ctx.direction == 'bullish':
        fit += 25
        reasons.append("Bullish bias, sell volatility above target")
    if ctx.iv_regime == 'high':
        fit += 20
    fit -= 15  # Penalize — undefined upside risk

    max_profit = (width + max(0, -net_per_share)) * ctx.lot_size  # At OTM strike
    return StrategyResult(
        name="Ratio Spread",
        category="ratio",
        legs=[
            StrategyLeg('BUY',  1, 'CALL', atm.strike, atm.call_ltp),
            StrategyLeg('SELL', 2, 'CALL', otm.strike, otm.call_ltp),
        ],
        net_debit=cost,
        max_profit=max_profit,
        max_loss=None,  # Unlimited above 2nd breakeven
        breakevens=[otm.strike + max_profit / ctx.lot_size],
        lot_size=ctx.lot_size,
        capital_required=otm.strike * ctx.lot_size * 0.2,
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Asymmetric bullish",
        risk_profile='undefined',
        direction_bias='bullish',
    )


def build_backspread(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Call Backspread (Long): Sell 1 ATM call, Buy 2 OTM calls."""
    atm = _find_atm(ctx.strikes, ctx.spot_price)
    otm = _find_strike_at_distance(ctx.strikes, ctx.spot_price, target_distance_pct=5)
    if not atm or not otm or atm.strike >= otm.strike: return None
    if atm.call_ltp <= 0 or otm.call_ltp <= 0: return None

    net_per_share = 2 * otm.call_ltp - atm.call_ltp  # Debit if 2*OTM > ATM (usually credit)
    cost = net_per_share * ctx.lot_size

    fit = 0
    reasons = []
    if ctx.direction == 'bullish':
        fit += 30
        reasons.append("Strong bullish with vol expansion")
    if ctx.iv_regime == 'low':
        fit += 25
        reasons.append("Low IV — long extra calls cheap")
    if ctx.conviction == 'strong': fit += 15

    return StrategyResult(
        name="Backspread",
        category="ratio",
        legs=[
            StrategyLeg('SELL', 1, 'CALL', atm.strike, atm.call_ltp),
            StrategyLeg('BUY',  2, 'CALL', otm.strike, otm.call_ltp),
        ],
        net_debit=cost,
        max_profit=None,
        max_loss=(otm.strike - atm.strike + max(0, net_per_share)) * ctx.lot_size,
        breakevens=[otm.strike + (otm.strike - atm.strike) + max(0, net_per_share)],
        lot_size=ctx.lot_size,
        capital_required=max(cost, 0) + (otm.strike * ctx.lot_size * 0.1),
        fit_score=max(0, min(100, fit)),
        fit_reason="; ".join(reasons) or "Backspread bullish",
        risk_profile='defined',
        direction_bias='bullish',
    )


def build_calendar_spread(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Calendar Spread: Sell near-term ATM, Buy far-term ATM (same strike)."""
    # Without multiple expiry chains, we can't truly price this.
    # Return informational result with disclaimer.
    return None  # Placeholder — needs multi-expiry data


def build_diagonal_spread(ctx: StrategyContext) -> Optional[StrategyResult]:
    """Diagonal: Different strikes AND different expiries."""
    return None  # Same limitation as calendar


# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY: List[Callable[[StrategyContext], Optional[StrategyResult]]] = [
    build_long_call,
    build_long_put,
    build_covered_call,
    build_covered_put,
    build_bull_call_spread,
    build_bear_put_spread,
    build_bull_put_spread,
    build_bear_call_spread,
    build_protective_put,
    build_collar,
    build_long_straddle,
    build_short_straddle,
    build_long_strangle,
    build_short_strangle,
    build_iron_condor,
    build_iron_butterfly,
    build_butterfly_spread,
    build_ratio_spread,
    build_backspread,
    build_jade_lizard,
    build_reverse_iron_condor,
    build_synthetic_long,
    build_synthetic_short,
    build_covered_strangle,
    build_strip,
    build_strap,
    # Excluded (need richer data than what we have):
    # build_calendar_spread,
    # build_diagonal_spread,
    # build_christmas_tree,
    # Excluded by design (not single-position strategies):
    # Gamma Scalping, Delta Neutral, Wheel Strategy, Box Spread
]


def build_all_strategies(ctx: StrategyContext) -> List[StrategyResult]:
    """Run every strategy builder, return non-None results sorted by fit_score."""
    results = []
    for builder in STRATEGY_REGISTRY:
        try:
            r = builder(ctx)
            if r is not None:
                results.append(r)
        except Exception:
            # Skip failed strategies silently
            pass
    results.sort(key=lambda r: r.fit_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Outlook derivation (from existing data)
# ---------------------------------------------------------------------------
def derive_outlook(price_data: dict, options_data, analyst_data: dict,
                   iv_context: Optional[dict] = None) -> dict:
    """
    Convert technical/options/analyst signals into outlook flags used by
    the strategy engine. Returns dict with direction, conviction, iv_regime.

    iv_context (added 2026): optional dict from _build_iv_context() in app.py
    containing real ATM IV and India VIX percentile. When provided, takes
    PRIORITY over the legacy PCR-based IV regime proxy.

    Expected iv_context shape:
      {"atm_iv": 0.25, "vix_percentile": 65.0, "vix_regime": "elevated", ...}
    """
    # ---- Direction ----
    bullish_signals = 0
    bearish_signals = 0

    # Price above MAs = bullish
    if price_data.get("current_price", 0) > price_data.get("sma_50", 0):  bullish_signals += 1
    else:                                                                  bearish_signals += 1
    if price_data.get("current_price", 0) > price_data.get("sma_200", 0): bullish_signals += 1
    else:                                                                  bearish_signals += 1

    # MACD
    if price_data.get("macd_hist", 0) > 0: bullish_signals += 1
    else:                                   bearish_signals += 1

    # RSI
    rsi = price_data.get("rsi_14", 50)
    if rsi > 55: bullish_signals += 1
    elif rsi < 45: bearish_signals += 1

    # Options composite signal
    if options_data:
        sig = options_data.composite_signal.lower() if hasattr(options_data, 'composite_signal') else ""
        if 'bullish' in sig: bullish_signals += 2
        if 'bearish' in sig: bearish_signals += 2

    # Analyst consensus
    consensus = (analyst_data or {}).get("consensus", "").lower()
    if 'buy' in consensus:  bullish_signals += 1
    if 'sell' in consensus: bearish_signals += 1

    # Decide direction
    diff = bullish_signals - bearish_signals
    if   diff >= 3:  direction = 'bullish'
    elif diff <= -3: direction = 'bearish'
    else:            direction = 'neutral'

    # ---- Conviction (how strong is the signal) ----
    total = bullish_signals + bearish_signals
    dominance = abs(diff) / max(total, 1)
    if   dominance >= 0.5:  conviction = 'strong'
    elif dominance >= 0.3:  conviction = 'moderate'
    else:                    conviction = 'weak'

    # ---- IV Regime ----
    # Tier 1: use real IV context if provided (VIX percentile is the most
    # reliable signal for "are options expensive right now")
    iv_regime = 'normal'
    if iv_context and iv_context.get("vix_percentile") is not None:
        pct = iv_context["vix_percentile"]
        if pct >= 70:   iv_regime = 'high'   # premiums expensive
        elif pct <= 30: iv_regime = 'low'    # premiums cheap
        else:            iv_regime = 'normal'
    elif iv_context and iv_context.get("atm_iv") is not None:
        # Fallback: per-stock ATM IV without historical context.
        # Use absolute thresholds calibrated to Indian single-stock IVs
        # (typically 15-35% normal, >40% elevated, <15% unusually low)
        atm_iv = iv_context["atm_iv"]
        if atm_iv >= 0.40:   iv_regime = 'high'
        elif atm_iv <= 0.15: iv_regime = 'low'
        else:                 iv_regime = 'normal'
    elif options_data:
        # Tier 3: legacy PCR proxy (kept for back-compat when iv_context
        # isn't available — e.g., if VIX fetch failed and no ATM IV computed)
        pcr = getattr(options_data, 'pcr_oi', 1.0)
        composite_strength = getattr(options_data, 'composite_strength', 0)
        if pcr > 1.3 or pcr < 0.7 or composite_strength >= 4:
            iv_regime = 'high'
        elif 0.9 <= pcr <= 1.1 and composite_strength <= 2:
            iv_regime = 'low'

    return {
        'direction': direction,
        'conviction': conviction,
        'iv_regime': iv_regime,
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
    }
