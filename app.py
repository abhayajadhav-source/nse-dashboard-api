"""
NSE Dashboard API — read-only JSON for the Cloudflare Pages dashboard,
plus on-demand stock analysis with Anthropic Claude, options analysis (Upstox),
and analyst ratings with Postgres caching.

Endpoints:
  GET  /              → health check
  GET  /api/snapshot  → latest scanner snapshots
  POST /api/analyze   → full AI analysis (technicals + analyst + news + options)
  POST /api/options   → standalone options-only analysis
  POST /api/compare   → side-by-side comparison of 2 stocks with AI verdict
  POST /api/strategy  → AI-recommended options strategy with concrete legs + P/L
  POST /api/position  → AI guidance on an existing position (hold/exit/scale)
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import pandas as pd
import psycopg2
import yfinance as yf
from anthropic import Anthropic
from flask import Flask, jsonify, request
from flask_cors import CORS

from analyst_cache import get_cached, save_cache
from options_analyzer import OptionChainData, fetch_options_data
from strategy_engine import (
    StrategyContext, OptionStrike,
    build_all_strategies, derive_outlook,
)
from technical_indicators import atr, bollinger_bands, ema, macd, rsi, sma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

DATABASE_URL      = os.getenv("DATABASE_URL", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-haiku-4-5"

anthropic_client: Optional[Anthropic] = None
if ANTHROPIC_API_KEY:
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------
@contextmanager
def _conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    c = psycopg2.connect(DATABASE_URL)
    try:
        yield c
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Health + snapshot
# ---------------------------------------------------------------------------
@app.route("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "nse-dashboard-api",
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "upstox_configured":    bool(os.getenv("UPSTOX_ACCESS_TOKEN", "")),
    }), 200


@app.route("/api/snapshot")
def get_snapshot():
    try:
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute("SELECT report_type, updated_at, payload FROM snapshots")
                out = {}
                for report_type, updated_at, payload in cur.fetchall():
                    out[report_type] = {
                        "updated_at": updated_at.isoformat(),
                        **payload,
                    }
                return jsonify(out), 200
    except psycopg2.errors.UndefinedTable:
        return jsonify({}), 200
    except Exception as e:
        logger.exception("Snapshot query failed: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Price data + indicators
# ---------------------------------------------------------------------------
def _fetch_price_data(symbol: str) -> Optional[dict]:
    """Fetch 6mo of daily data + compute all indicators for a stock."""
    yf_symbol = f"{symbol}.NS"
    try:
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="6mo", interval="1d")
        if hist.empty or len(hist) < 60:
            return None

        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]

        sma_20  = sma(close, 20)
        sma_50  = sma(close, 50)
        sma_200 = sma(close, 200) if len(close) >= 200 else pd.Series([float("nan")] * len(close), index=close.index)
        ema_9   = ema(close, 9)
        ema_21  = ema(close, 21)
        rsi_14  = rsi(close, 14)
        macd_line, signal_line, histogram = macd(close, 12, 26, 9)
        bb_upper, bb_middle, bb_lower = bollinger_bands(close, 20, 2.0)
        atr_14  = atr(high, low, close, 14)

        last_252 = hist.iloc[-252:] if len(hist) >= 252 else hist
        high_52w = float(last_252["High"].max())
        low_52w  = float(last_252["Low"].min())

        last = close.iloc[-1]
        prev = close.iloc[-2]
        pct_1d = ((last - prev) / prev) * 100

        vol_20d_avg = volume.iloc[-20:].mean()
        vol_today   = volume.iloc[-1]
        vol_ratio   = vol_today / vol_20d_avg if vol_20d_avg > 0 else 0

        def last_val(series, default=0.0):
            try:
                v = series.iloc[-1]
                return float(v) if pd.notna(v) else default
            except Exception:
                return default

        return {
            "symbol":          symbol,
            "current_price":   float(last),
            "prev_close":      float(prev),
            "pct_change_1d":   round(pct_1d, 2),
            "day_high":        float(high.iloc[-1]),
            "day_low":         float(low.iloc[-1]),
            "volume":          int(vol_today),
            "volume_20d_avg":  int(vol_20d_avg),
            "volume_ratio":    round(vol_ratio, 2),

            "sma_20":      round(last_val(sma_20),   2),
            "sma_50":      round(last_val(sma_50),   2),
            "sma_200":     round(last_val(sma_200),  2),
            "ema_9":       round(last_val(ema_9),    2),
            "ema_21":      round(last_val(ema_21),   2),

            "rsi_14":      round(last_val(rsi_14, 50.0), 1),
            "macd_line":   round(last_val(macd_line),    3),
            "macd_signal": round(last_val(signal_line),  3),
            "macd_hist":   round(last_val(histogram),    3),

            "bb_upper":   round(last_val(bb_upper),  2),
            "bb_middle":  round(last_val(bb_middle), 2),
            "bb_lower":   round(last_val(bb_lower),  2),
            "atr_14":     round(last_val(atr_14),    2),

            "high_52w":   round(high_52w, 2),
            "low_52w":    round(low_52w, 2),
            "pct_from_52w_high": round((last - high_52w) / high_52w * 100, 2),
            "pct_from_52w_low":  round((last - low_52w)  / low_52w  * 100, 2),

            "pct_change_5d":  round(((last - close.iloc[-6])  / close.iloc[-6])  * 100, 2) if len(close) > 5 else 0.0,
            "pct_change_30d": round(((last - close.iloc[-21]) / close.iloc[-21]) * 100, 2) if len(close) > 20 else 0.0,
            "pct_change_90d": round(((last - close.iloc[-63]) / close.iloc[-63]) * 100, 2) if len(close) > 62 else 0.0,
        }
    except Exception as e:
        logger.exception("Price fetch failed for %s: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# Analyst ratings — now CACHE-AWARE
# ---------------------------------------------------------------------------
def _fetch_analyst_ratings(symbol: str) -> dict:
    """
    Fetch analyst recommendations and price targets.

    Cache-first strategy:
      1. Check Postgres cache — if fresh (<6h), return immediately
      2. Otherwise, call yfinance with multiple fallback strategies
      3. On success, save to cache and return fresh data
      4. On rate-limit / failure, return empty result (cold-start behavior:
         show 'no data' rather than ancient cached entries)

    Known issue: yfinance's `ticker.info` for .NS (Indian) symbols often
    returns a partial response WITHOUT the analyst fields, even though
    Yahoo's website shows them. We log what's missing for diagnostics, and
    try multiple yfinance attributes as fallbacks before giving up.
    """
    yf_symbol = f"{symbol}.NS"
    empty_result = {
        "summary": None, "consensus": None, "price_target": None,
        "recent_actions": [], "data_source": "unavailable",
    }

    # ---- Step 1: Try cache first ----
    cached = get_cached(symbol)
    if cached is not None:
        return cached

    # ---- Step 2: Cache miss → fetch from yfinance ----
    result = {
        "summary": None, "consensus": None, "price_target": None,
        "recent_actions": [], "data_source": "unavailable",
    }

    try:
        ticker = yf.Ticker(yf_symbol)
        info   = ticker.info or {}

        # Diagnostic: log what analyst-related keys are actually present in info.
        # If you see "[no analyst keys]" repeatedly, Yahoo's .NS feed is the issue.
        analyst_keys_present = [
            k for k in info.keys()
            if any(w in k.lower() for w in ['recommend', 'analyst', 'target'])
        ]
        if analyst_keys_present:
            logger.info("Analyst keys present for %s: %s", symbol, analyst_keys_present)
        else:
            logger.warning("No analyst keys in ticker.info for %s (info had %d total keys) — "
                           "Yahoo .NS feed limitation", symbol, len(info))

        rec_key   = info.get("recommendationKey")
        rec_mean  = info.get("recommendationMean")
        num_anlst = info.get("numberOfAnalystOpinions")

        # --- Strategy 1: recommendations_summary attribute (aggregated counts) ---
        try:
            rec_df = ticker.recommendations_summary
            if rec_df is not None and not rec_df.empty:
                row = rec_df.iloc[0]
                result["summary"] = {
                    "strong_buy":  int(row.get("strongBuy",  0) or 0),
                    "buy":         int(row.get("buy",        0) or 0),
                    "hold":        int(row.get("hold",       0) or 0),
                    "sell":        int(row.get("sell",       0) or 0),
                    "strong_sell": int(row.get("strongSell", 0) or 0),
                }
                result["summary"]["total"] = sum(result["summary"].values())
                logger.info("Got recommendations_summary for %s: %d analysts",
                            symbol, result["summary"]["total"])
        except Exception as e:
            logger.debug("recommendations_summary failed for %s: %s", symbol, e)

        # --- Strategy 2: legacy recommendations attribute (raw broker actions) ---
        # If summary failed, try the older `recommendations` attribute. It's a
        # different shape (history of broker actions) but we can derive a summary
        # from the most recent N entries.
        if result["summary"] is None:
            try:
                old_rec = ticker.recommendations
                if old_rec is not None and not old_rec.empty:
                    # Get the most recent ~20 actions and tally them
                    recent_actions_df = old_rec.tail(20)
                    counts = {"strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0}
                    for _, row in recent_actions_df.iterrows():
                        grade = str(row.get("To Grade", "") or row.get("ToGrade", "")).lower().strip()
                        if "strong buy" in grade or "strongbuy" in grade:
                            counts["strong_buy"] += 1
                        elif "buy" in grade or "outperform" in grade or "overweight" in grade:
                            counts["buy"] += 1
                        elif "hold" in grade or "neutral" in grade or "market perform" in grade:
                            counts["hold"] += 1
                        elif "strong sell" in grade or "strongsell" in grade:
                            counts["strong_sell"] += 1
                        elif "sell" in grade or "underperform" in grade or "underweight" in grade:
                            counts["sell"] += 1
                    total = sum(counts.values())
                    if total > 0:
                        counts["total"] = total
                        result["summary"] = counts
                        logger.info("Derived summary from recommendations history for %s: %d actions",
                                    symbol, total)
            except Exception as e:
                logger.debug("recommendations history failed for %s: %s", symbol, e)

        # --- Strategy 3: fall back to just analyst count from info ---
        if result["summary"] is None and num_anlst:
            result["summary"] = {
                "strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0,
                "total": int(num_anlst),
            }
            logger.info("Got analyst count only for %s: %d (no breakdown)", symbol, num_anlst)

        # --- Consensus label ---
        if rec_key and rec_key != "none":
            label_map = {
                "strong_buy": "Strong Buy", "buy": "Buy", "hold": "Hold",
                "sell": "Sell", "strong_sell": "Strong Sell",
                "underperform": "Underperform", "outperform": "Outperform",
            }
            result["consensus"] = label_map.get(rec_key, rec_key.replace("_", " ").title())
        elif rec_mean:
            if   rec_mean <= 1.5: result["consensus"] = "Strong Buy"
            elif rec_mean <= 2.5: result["consensus"] = "Buy"
            elif rec_mean <= 3.5: result["consensus"] = "Hold"
            elif rec_mean <= 4.5: result["consensus"] = "Sell"
            else:                 result["consensus"] = "Strong Sell"

        # --- Price targets ---
        target_mean   = info.get("targetMeanPrice")
        target_high   = info.get("targetHighPrice")
        target_low    = info.get("targetLowPrice")
        target_median = info.get("targetMedianPrice")
        target_count  = info.get("numberOfAnalystOpinions")
        currency      = info.get("currency", "INR")

        if target_mean:
            result["price_target"] = {
                "mean":     round(float(target_mean), 2),
                "median":   round(float(target_median), 2) if target_median else None,
                "high":     round(float(target_high), 2)   if target_high   else None,
                "low":      round(float(target_low), 2)    if target_low    else None,
                "count":    int(target_count) if target_count else None,
                "currency": currency,
            }
            logger.info("Got price target for %s: ₹%.2f (n=%s)",
                        symbol, target_mean, target_count)

        # --- Strategy 4: analyst_price_targets (newer yfinance attribute) ---
        # If we didn't get price target from info, try the dedicated method
        if result["price_target"] is None:
            try:
                apt = ticker.analyst_price_targets
                if apt and isinstance(apt, dict) and apt.get("mean"):
                    result["price_target"] = {
                        "mean":     round(float(apt["mean"]), 2),
                        "median":   round(float(apt["median"]), 2) if apt.get("median") else None,
                        "high":     round(float(apt["high"]), 2) if apt.get("high") else None,
                        "low":      round(float(apt["low"]), 2) if apt.get("low") else None,
                        "count":    None,
                        "currency": currency,
                    }
                    logger.info("Got price target via analyst_price_targets for %s: ₹%.2f",
                                symbol, apt["mean"])
            except Exception as e:
                logger.debug("analyst_price_targets failed for %s: %s", symbol, e)

        # --- Recent broker upgrades/downgrades ---
        try:
            upgrades_df = ticker.upgrades_downgrades
            if upgrades_df is not None and not upgrades_df.empty:
                recent = upgrades_df.head(8)
                for idx, row in recent.iterrows():
                    try:
                        action_date = pd.to_datetime(idx).strftime("%d %b %Y")
                    except Exception:
                        action_date = str(idx)[:10]
                    result["recent_actions"].append({
                        "date":       action_date,
                        "firm":       str(row.get("Firm", "") or ""),
                        "to_grade":   str(row.get("ToGrade", "") or ""),
                        "from_grade": str(row.get("FromGrade", "") or ""),
                        "action":     str(row.get("Action", "") or ""),
                    })
                if result["recent_actions"]:
                    logger.info("Got %d recent actions for %s",
                                len(result["recent_actions"]), symbol)
        except Exception as e:
            logger.debug("upgrades_downgrades failed for %s: %s", symbol, e)

        if (result["summary"] or result["consensus"] or
            result["price_target"] or result["recent_actions"]):
            result["data_source"] = "yfinance"
        else:
            logger.warning(
                "All analyst strategies returned empty for %s — "
                "Yahoo doesn't have data for this stock, or .NS feed is partial",
                symbol
            )

    except Exception as e:
        # Most common case: Yahoo "Too Many Requests" rate limit
        # Return empty result; do NOT save to cache (so next request retries)
        logger.warning("Analyst rating fetch failed for %s: %s", symbol, e)
        return empty_result

    # ---- Step 3: Save successful fetch to cache for next time ----
    save_cache(symbol, result)

    return result


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
def _fetch_news(symbol: str, max_items: int = 6) -> list:
    query = urllib.parse.quote_plus(f'"{symbol}" India stock')
    url = (
        f"https://news.google.com/rss/search?"
        f"q={query}+when:7d"
        f"&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        logger.warning("News fetch failed for %s: %s", symbol, e)
        return []

    if not feed.entries:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    items  = []
    for entry in feed.entries[:max_items * 2]:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        else:
            published = datetime.now(timezone.utc)
        if published < cutoff:
            continue

        title = getattr(entry, "title", "").strip()
        if not title:
            continue

        source = title.rsplit(" - ", 1)[-1] if " - " in title else "Unknown"
        clean_title = title.rsplit(" - ", 1)[0] if " - " in title else title

        items.append({
            "title":     clean_title,
            "link":      getattr(entry, "link", ""),
            "source":    source,
            "published": published.isoformat(),
            "age_days":  (datetime.now(timezone.utc) - published).days,
        })
        if len(items) >= max_items:
            break

    return items


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _format_ratings_for_prompt(ratings: dict, current_price: float) -> str:
    if ratings.get("data_source") == "unavailable":
        return "  (no analyst data available)"
    lines = []
    if ratings.get("consensus"):
        lines.append(f"  Consensus rating: {ratings['consensus']}")
    s = ratings.get("summary")
    if s and s.get("total", 0) > 0:
        if any([s["strong_buy"], s["buy"], s["hold"], s["sell"], s["strong_sell"]]):
            lines.append(
                f"  Breakdown: Strong Buy {s['strong_buy']} · Buy {s['buy']} · "
                f"Hold {s['hold']} · Sell {s['sell']} · Strong Sell {s['strong_sell']} "
                f"(total {s['total']} analysts)"
            )
    pt = ratings.get("price_target")
    if pt and pt.get("mean"):
        upside = ((pt["mean"] - current_price) / current_price) * 100
        target_str = f"  Price target (consensus): ₹{pt['mean']:,.2f}"
        if pt.get("high") and pt.get("low"):
            target_str += f" (range ₹{pt['low']:,.2f} – ₹{pt['high']:,.2f})"
        target_str += f" → {upside:+.1f}% from current ₹{current_price:,.2f}"
        lines.append(target_str)
    actions = ratings.get("recent_actions", [])
    if actions:
        lines.append(f"  Recent broker actions ({len(actions)}):")
        for a in actions[:5]:
            transition = ""
            if a.get("from_grade") and a.get("to_grade") and a["from_grade"] != a["to_grade"]:
                transition = f" ({a['from_grade']} → {a['to_grade']})"
            elif a.get("to_grade"):
                transition = f" ({a['to_grade']})"
            lines.append(f"    {a['date']} — {a['firm']}{transition}")
    return "\n".join(lines) if lines else "  (no analyst data)"


def _format_options_for_prompt(opts: Optional[OptionChainData]) -> str:
    """Compact text block summarising options data for Claude."""
    if opts is None:
        return "  (no options data available — likely not an F&O stock)"

    lines = []
    lines.append(f"  Expiry analysed: {opts.expiry_date}")
    lines.append(f"  PCR (OI): {opts.pcr_oi:.2f} → {opts.pcr_signal}")
    if opts.pcr_volume > 0:
        lines.append(f"  PCR (Volume): {opts.pcr_volume:.2f}")
    if opts.max_pain > 0:
        pct_from_pain = ((opts.underlying_price - opts.max_pain) / opts.max_pain) * 100
        lines.append(
            f"  Max Pain: ₹{opts.max_pain:.2f} "
            f"(price is {pct_from_pain:+.2f}% from max pain)"
        )
    lines.append(
        f"  Highest Call OI strike: ₹{opts.highest_call_oi_strike:.2f} "
        "(acts as resistance — call writers defend this)"
    )
    lines.append(
        f"  Highest Put OI strike:  ₹{opts.highest_put_oi_strike:.2f} "
        "(acts as support — put writers defend this)"
    )
    lines.append(
        f"  Day's OI change: Calls {opts.total_call_oi_change:+,} · "
        f"Puts {opts.total_put_oi_change:+,} → {opts.buildup_signal}"
    )
    lines.append(
        f"  Composite signal: {opts.composite_signal} "
        f"(strength {opts.composite_strength}/5)"
    )
    if opts.confirmations:
        lines.append("  Signal rationale:")
        for c in opts.confirmations:
            lines.append(f"    - {c}")
    return "\n".join(lines)


def _build_analysis_prompt(price_data: dict, news: list, ratings: dict,
                          options: Optional[OptionChainData]) -> str:
    p = price_data

    news_block = ""
    if news:
        news_lines = []
        for n in news:
            age = "today" if n["age_days"] == 0 else f"{n['age_days']}d ago"
            news_lines.append(f"  - [{age}] {n['title']} ({n['source']})")
        news_block = "\n".join(news_lines)
    else:
        news_block = "  (no recent headlines)"

    ratings_block = _format_ratings_for_prompt(ratings, p["current_price"])
    options_block = _format_options_for_prompt(options)

    return f"""You are a technical analyst for Indian (NSE) stocks. Analyze the following daily-timeframe data for {p['symbol']} and provide a balanced, actionable assessment.

PRICE & VOLUME
- Current price: ₹{p['current_price']:,.2f}
- Today's change: {p['pct_change_1d']:+.2f}%
- 5d/30d/90d changes: {p['pct_change_5d']:+.2f}% / {p['pct_change_30d']:+.2f}% / {p['pct_change_90d']:+.2f}%
- Today's range: ₹{p['day_low']:,.2f} – ₹{p['day_high']:,.2f}
- Volume today: {p['volume']:,} (vs 20d avg {p['volume_20d_avg']:,} → {p['volume_ratio']}x)

52-WEEK RANGE
- 52w high: ₹{p['high_52w']:,.2f} ({p['pct_from_52w_high']:+.2f}% from high)
- 52w low:  ₹{p['low_52w']:,.2f}  ({p['pct_from_52w_low']:+.2f}% from low)

TREND (Moving Averages)
- SMA 20:  ₹{p['sma_20']:,.2f}  ({'above' if p['current_price'] > p['sma_20']  else 'below'})
- SMA 50:  ₹{p['sma_50']:,.2f}  ({'above' if p['current_price'] > p['sma_50']  else 'below'})
- SMA 200: ₹{p['sma_200']:,.2f} ({'above' if p['current_price'] > p['sma_200'] else 'below'})
- EMA 9 / EMA 21: ₹{p['ema_9']:,.2f} / ₹{p['ema_21']:,.2f}

MOMENTUM
- RSI (14): {p['rsi_14']} {'(overbought)' if p['rsi_14'] > 70 else '(oversold)' if p['rsi_14'] < 30 else '(neutral)'}
- MACD line / signal / histogram: {p['macd_line']} / {p['macd_signal']} / {p['macd_hist']}

VOLATILITY
- Bollinger upper/middle/lower: ₹{p['bb_upper']:,.2f} / ₹{p['bb_middle']:,.2f} / ₹{p['bb_lower']:,.2f}
- ATR (14): ₹{p['atr_14']:,.2f}

OPTIONS DATA (F&O smart-money positioning)
{options_block}

ANALYST RATINGS & PRICE TARGETS
{ratings_block}

RECENT NEWS (last 7 days)
{news_block}

INSTRUCTIONS
Provide your analysis in exactly this structure, using Markdown:

## Trend Summary
2-3 sentences describing trend direction, strength, conviction.

## Key Technical Observations
- 4-6 bullets covering MA structure, RSI/MACD, volatility, volume
- Be specific with numbers

## Options Sentiment Read
- What is F&O smart money signalling?
- Tie PCR + OI buildup + max pain into a coherent view
- Specifically address: does options positioning agree or conflict with the technicals?
- Note key levels from OI: highest call OI = resistance, highest put OI = support
- If no options data, say so and skip this section

## Support & Resistance
- Immediate support: ₹X (rationale — include options OI levels if relevant)
- Immediate resistance: ₹Y (rationale)
- Use price pivots, MAs, BB, and OI levels

## Analyst View vs Technical View
- Compare broker rating + target with technical setup
- Flag disagreements
- If no analyst data, say so and skip this section

## News Context
1-2 sentences linking news to the setup.

## Recommendation
One of: **STRONG BUY**, **BUY**, **HOLD**, **SELL**, **STRONG SELL**, **AVOID**

Follow with:
- Time horizon (intraday / swing 1-4 weeks / positional 1-3 months)
- Entry zone / Stop-loss
- Target (use both technical levels and options-implied levels)
- Risk-reward

## Risks & Caveats
2-3 bullets on what invalidates the view.

GUARDRAILS
- Honest about uncertainty.
- When options data conflicts with technicals, explicitly say which one you weight more and why.
- Treat options data as one input, not a single magic indicator.
- End with: "*Technical analysis only. Not investment advice.*"
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.route("/api/options", methods=["POST"])
def get_options():
    """Standalone options-only analysis (faster — no AI call)."""
    data   = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    price_data = _fetch_price_data(symbol)
    if price_data is None:
        return jsonify({"error": f"Could not fetch price data for {symbol}"}), 404

    opts = fetch_options_data(symbol, price_data["current_price"])
    if opts is None:
        return jsonify({
            "error": "no_fo_data",
            "message": f"{symbol} appears not to be an F&O stock, "
                       "or option chain data is temporarily unavailable. "
                       "Try a NIFTY 50 stock like RELIANCE or HDFCBANK."
        }), 404

    return jsonify({
        "symbol":           opts.symbol,
        "underlying_price": opts.underlying_price,
        "expiry_date":      opts.expiry_date,
        "pcr_oi":           round(opts.pcr_oi, 2),
        "pcr_volume":       round(opts.pcr_volume, 2),
        "max_pain":         round(opts.max_pain, 2),
        "pct_from_max_pain": round((opts.underlying_price - opts.max_pain) / opts.max_pain * 100, 2) if opts.max_pain else 0,
        "highest_call_oi_strike": opts.highest_call_oi_strike,
        "highest_put_oi_strike":  opts.highest_put_oi_strike,
        "total_call_oi":          opts.total_call_oi,
        "total_put_oi":           opts.total_put_oi,
        "total_call_oi_change":   opts.total_call_oi_change,
        "total_put_oi_change":    opts.total_put_oi_change,
        "pcr_signal":             opts.pcr_signal,
        "buildup_signal":         opts.buildup_signal,
        "composite_signal":       opts.composite_signal,
        "composite_strength":     opts.composite_strength,
        "confirmations":          opts.confirmations,
        "top_strikes":            opts.top_strikes,
    }), 200


@app.route("/api/analyze", methods=["POST"])
def analyze_stock():
    if not anthropic_client:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    data   = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    if not all(c.isalnum() or c in "&-" for c in symbol):
        return jsonify({"error": "invalid symbol"}), 400

    logger.info("Analyzing %s", symbol)

    price_data = _fetch_price_data(symbol)
    if price_data is None:
        return jsonify({"error": f"Could not fetch data for {symbol}"}), 404

    ratings = _fetch_analyst_ratings(symbol)
    news    = _fetch_news(symbol)
    options = fetch_options_data(symbol, price_data["current_price"])   # may be None for non-F&O

    prompt = _build_analysis_prompt(price_data, news, ratings, options)
    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
    except Exception as e:
        logger.exception("Anthropic API call failed: %s", e)
        return jsonify({"error": f"AI analysis failed: {str(e)}"}), 500

    # Serialize options data for the dashboard
    options_payload = None
    if options is not None:
        options_payload = {
            "symbol":              options.symbol,
            "expiry_date":         options.expiry_date,
            "pcr_oi":              round(options.pcr_oi, 2),
            "pcr_volume":          round(options.pcr_volume, 2),
            "max_pain":            round(options.max_pain, 2),
            "pct_from_max_pain":   round((options.underlying_price - options.max_pain) / options.max_pain * 100, 2) if options.max_pain else 0,
            "highest_call_oi_strike": options.highest_call_oi_strike,
            "highest_put_oi_strike":  options.highest_put_oi_strike,
            "total_call_oi":          options.total_call_oi,
            "total_put_oi":           options.total_put_oi,
            "total_call_oi_change":   options.total_call_oi_change,
            "total_put_oi_change":    options.total_put_oi_change,
            "pcr_signal":             options.pcr_signal,
            "buildup_signal":         options.buildup_signal,
            "composite_signal":       options.composite_signal,
            "composite_strength":     options.composite_strength,
            "confirmations":          options.confirmations,
            "top_strikes":            options.top_strikes,
        }

    return jsonify({
        "symbol":      symbol,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "price_data":  price_data,
        "ratings":     ratings,
        "options":     options_payload,
        "news":        news,
        "analysis":    analysis_text,
        "model":       ANTHROPIC_MODEL,
    }), 200


# ===========================================================================
# COMPARISON FEATURE — side-by-side comparison of 2 stocks
# ===========================================================================

def _fetch_stock_bundle(symbol: str) -> dict:
    """
    Fetch everything needed to compare ONE stock.

    Runs price first (options analyzer needs current price), then fans out
    to fetch analyst ratings, options data, and news in parallel.
    Returns a dict with keys: symbol, price, ratings, options, news.
    """
    bundle = {"symbol": symbol, "price": None, "ratings": None,
              "options": None, "news": []}

    bundle["price"] = _fetch_price_data(symbol)
    if bundle["price"] is None:
        return bundle  # caller checks for this and 404s

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_ratings = executor.submit(_fetch_analyst_ratings, symbol)
        future_options = executor.submit(
            fetch_options_data, symbol, bundle["price"]["current_price"]
        )
        future_news    = executor.submit(_fetch_news, symbol, 3)

        bundle["ratings"] = future_ratings.result()
        bundle["options"] = future_options.result()
        bundle["news"]    = future_news.result()

    return bundle


def _compute_comparison_metrics(a: dict, b: dict) -> dict:
    """
    Build the metric-by-metric comparison table + composite scores.
    Each row has a winner: 'a', 'b', 'tie', or 'na' (not applicable).
    """
    rows = []
    pa = a.get("price")
    pb = b.get("price")
    ra = a.get("ratings") or {}
    rb = b.get("ratings") or {}
    oa = a.get("options")
    ob = b.get("options")

    def add_row(section, metric, a_val, b_val, a_disp, b_disp, winner, reason=""):
        rows.append({
            "section": section, "metric": metric,
            "a_value": a_val, "b_value": b_val,
            "a_display": a_disp, "b_display": b_disp,
            "winner": winner, "reason": reason,
        })

    # ---- Section 1: Price & Momentum ----
    if pa and pb:
        a_1d, b_1d = pa["pct_change_1d"], pb["pct_change_1d"]
        add_row("Price & Momentum", "Today's change",
                a_1d, b_1d, f"{a_1d:+.2f}%", f"{b_1d:+.2f}%",
                "a" if a_1d > b_1d else "b" if b_1d > a_1d else "tie",
                "Larger same-day gain favours stronger momentum")

        a_30d, b_30d = pa["pct_change_30d"], pb["pct_change_30d"]
        add_row("Price & Momentum", "30-day return",
                a_30d, b_30d, f"{a_30d:+.2f}%", f"{b_30d:+.2f}%",
                "a" if a_30d > b_30d else "b" if b_30d > a_30d else "tie",
                "30-day return shows monthly trend strength")

        a_90d, b_90d = pa["pct_change_90d"], pb["pct_change_90d"]
        add_row("Price & Momentum", "90-day return",
                a_90d, b_90d, f"{a_90d:+.2f}%", f"{b_90d:+.2f}%",
                "a" if a_90d > b_90d else "b" if b_90d > a_90d else "tie",
                "Quarter-trend strength")

        a_vol, b_vol = pa["volume_ratio"], pb["volume_ratio"]
        add_row("Price & Momentum", "Volume vs 20d avg",
                a_vol, b_vol, f"{a_vol:.2f}x", f"{b_vol:.2f}x",
                "a" if a_vol > b_vol else "b" if b_vol > a_vol else "tie",
                "Higher volume = more institutional interest")

        # Position in 52w range — closer to high = stronger trend
        if pa["high_52w"] > pa["low_52w"]:
            a_pos = (pa["current_price"] - pa["low_52w"]) / (pa["high_52w"] - pa["low_52w"]) * 100
        else:
            a_pos = 50
        if pb["high_52w"] > pb["low_52w"]:
            b_pos = (pb["current_price"] - pb["low_52w"]) / (pb["high_52w"] - pb["low_52w"]) * 100
        else:
            b_pos = 50
        add_row("Price & Momentum", "Position in 52w range",
                round(a_pos, 1), round(b_pos, 1),
                f"{a_pos:.1f}% from low", f"{b_pos:.1f}% from low",
                "a" if a_pos > b_pos else "b" if b_pos > a_pos else "tie",
                "Stocks near 52w high show stronger uptrends")

    # ---- Section 2: Trend Indicators ----
    if pa and pb:
        # RSI quality: 50-65 is ideal, >75 or <30 is poor
        def rsi_quality(rsi):
            if 50 <= rsi <= 65: return 3
            if 40 <= rsi <= 50 or 65 < rsi <= 70: return 2
            if 30 <= rsi < 40 or 70 < rsi <= 75: return 1
            return 0

        a_rsi, b_rsi = pa["rsi_14"], pb["rsi_14"]
        a_rsi_q, b_rsi_q = rsi_quality(a_rsi), rsi_quality(b_rsi)
        a_rsi_label = "overbought" if a_rsi > 70 else "oversold" if a_rsi < 30 else "neutral"
        b_rsi_label = "overbought" if b_rsi > 70 else "oversold" if b_rsi < 30 else "neutral"
        add_row("Trend Indicators", "RSI (14)",
                a_rsi, b_rsi,
                f"{a_rsi} ({a_rsi_label})", f"{b_rsi} ({b_rsi_label})",
                "a" if a_rsi_q > b_rsi_q else "b" if b_rsi_q > a_rsi_q else "tie",
                "RSI in 50-65 zone = bullish but not overbought")

        def ma_count(p):
            return sum([p["current_price"] > p["sma_20"],
                        p["current_price"] > p["sma_50"],
                        p["current_price"] > p["sma_200"]])
        a_ma, b_ma = ma_count(pa), ma_count(pb)
        add_row("Trend Indicators", "Above MAs (20/50/200)",
                a_ma, b_ma, f"{a_ma}/3", f"{b_ma}/3",
                "a" if a_ma > b_ma else "b" if b_ma > a_ma else "tie",
                "More MAs below price = stronger uptrend")

        a_macd, b_macd = pa["macd_hist"], pb["macd_hist"]
        add_row("Trend Indicators", "MACD histogram",
                a_macd, b_macd,
                f"{a_macd:+.3f} ({'bullish' if a_macd > 0 else 'bearish'})",
                f"{b_macd:+.3f} ({'bullish' if b_macd > 0 else 'bearish'})",
                "a" if a_macd > b_macd else "b" if b_macd > a_macd else "tie",
                "Positive MACD histogram = upward momentum")

    # ---- Section 3: Options Sentiment ----
    if oa and ob:
        a_pcr, b_pcr = oa.pcr_oi, ob.pcr_oi
        add_row("Options Sentiment", "PCR (OI)",
                a_pcr, b_pcr,
                f"{a_pcr:.2f} ({oa.pcr_signal})",
                f"{b_pcr:.2f} ({ob.pcr_signal})",
                "a" if a_pcr > b_pcr else "b" if b_pcr > a_pcr else "tie",
                "Higher PCR = more put writing = bullish signal")

        def signal_score(signal: str, strength: int) -> int:
            s = signal.lower()
            if "strong bullish" in s:    return strength
            if "bullish" in s:           return max(strength - 1, 1)
            if "slightly bullish" in s:  return max(strength - 2, 1)
            if "strong bearish" in s:    return -strength
            if "bearish" in s:           return -max(strength - 1, 1)
            if "slightly bearish" in s:  return -max(strength - 2, 1)
            return 0

        a_sig = signal_score(oa.composite_signal, oa.composite_strength)
        b_sig = signal_score(ob.composite_signal, ob.composite_strength)
        add_row("Options Sentiment", "Composite signal",
                a_sig, b_sig,
                f"{oa.composite_signal} ({oa.composite_strength}/5)",
                f"{ob.composite_signal} ({ob.composite_strength}/5)",
                "a" if a_sig > b_sig else "b" if b_sig > a_sig else "tie",
                "Composite blends PCR + OI buildup + max pain")

        if oa.max_pain > 0 and ob.max_pain > 0:
            a_mp = (oa.underlying_price - oa.max_pain) / oa.max_pain * 100
            b_mp = (ob.underlying_price - ob.max_pain) / ob.max_pain * 100
            add_row("Options Sentiment", "% from max pain",
                    round(a_mp, 2), round(b_mp, 2),
                    f"{a_mp:+.2f}%", f"{b_mp:+.2f}%",
                    "a" if a_mp < b_mp else "b" if b_mp < a_mp else "tie",
                    "Price below max pain = pull-up bias")
    elif oa or ob:
        add_row("Options Sentiment", "F&O availability",
                bool(oa), bool(ob),
                "F&O available" if oa else "Not an F&O stock",
                "F&O available" if ob else "Not an F&O stock",
                "a" if oa and not ob else "b" if ob and not oa else "tie",
                "F&O stocks have more liquidity and signal quality")
    else:
        add_row("Options Sentiment", "F&O availability",
                False, False,
                "Not an F&O stock", "Not an F&O stock",
                "na", "Neither stock has options data")

    # ---- Section 4: Analyst Ratings ----
    if ra.get("data_source") == "yfinance" or rb.get("data_source") == "yfinance":
        def consensus_score(consensus):
            if not consensus: return 0
            c = consensus.lower()
            if "strong buy" in c:   return 3
            if "buy" in c:          return 2
            if "outperform" in c:   return 2
            if "hold" in c:         return 1
            if "underperform" in c: return -1
            if "strong sell" in c:  return -2
            if "sell" in c:         return -1
            return 0

        a_cons = consensus_score(ra.get("consensus"))
        b_cons = consensus_score(rb.get("consensus"))
        add_row("Analyst Ratings", "Consensus",
                a_cons, b_cons,
                ra.get("consensus") or "No data",
                rb.get("consensus") or "No data",
                "a" if a_cons > b_cons else "b" if b_cons > a_cons else (
                    "na" if a_cons == 0 and b_cons == 0 else "tie"),
                "Higher conviction rating from analysts")

        pt_a = (ra.get("price_target") or {}).get("mean")
        pt_b = (rb.get("price_target") or {}).get("mean")
        a_up = (pt_a - pa["current_price"]) / pa["current_price"] * 100 if (pt_a and pa) else None
        b_up = (pt_b - pb["current_price"]) / pb["current_price"] * 100 if (pt_b and pb) else None

        if a_up is not None or b_up is not None:
            add_row("Analyst Ratings", "Upside to target",
                    a_up if a_up is not None else 0,
                    b_up if b_up is not None else 0,
                    f"{a_up:+.1f}%" if a_up is not None else "No target",
                    f"{b_up:+.1f}%" if b_up is not None else "No target",
                    "a" if (a_up is not None and (b_up is None or a_up > b_up)) else
                    "b" if (b_up is not None and (a_up is None or b_up > a_up)) else "tie",
                    "Higher implied upside from broker price targets")

        n_a = ((ra.get("price_target") or {}).get("count") or
               (ra.get("summary") or {}).get("total") or 0)
        n_b = ((rb.get("price_target") or {}).get("count") or
               (rb.get("summary") or {}).get("total") or 0)
        add_row("Analyst Ratings", "# Analysts covering",
                n_a, n_b,
                f"{n_a} analysts" if n_a else "No coverage",
                f"{n_b} analysts" if n_b else "No coverage",
                "a" if n_a > n_b else "b" if n_b > n_a else "tie",
                "Broader coverage = more reliable consensus")
    else:
        add_row("Analyst Ratings", "Analyst data",
                False, False,
                "No coverage", "No coverage",
                "na", "Yahoo Finance has no analyst data for either")

    # ---- Section 5: Recent News (informational only) ----
    a_news = a.get("news") or []
    b_news = b.get("news") or []
    add_row("Recent News", "Headlines (last 7 days)",
            len(a_news), len(b_news),
            f"{len(a_news)} headlines", f"{len(b_news)} headlines",
            "na", "News headlines listed below for context")

    # ---- Composite scores ----
    scores = {"a": 0, "b": 0, "ties": 0, "na": 0}
    for row in rows:
        if   row["winner"] == "a":   scores["a"]    += 1
        elif row["winner"] == "b":   scores["b"]    += 1
        elif row["winner"] == "tie": scores["ties"] += 1
        else:                        scores["na"]   += 1

    verdict = "a" if scores["a"] > scores["b"] else "b" if scores["b"] > scores["a"] else "tie"

    return {"rows": rows, "scores": scores, "verdict": verdict}


def _build_comparison_prompt(a: dict, b: dict, metrics: dict) -> str:
    """Build Claude prompt asking for executive summary + case-for-each + verdict."""
    sym_a = a["symbol"]
    sym_b = b["symbol"]
    pa = a.get("price") or {}
    pb = b.get("price") or {}
    ra = a.get("ratings") or {}
    rb = b.get("ratings") or {}
    oa = a.get("options")
    ob = b.get("options")

    # Compact metric comparison table
    table_lines = []
    current_section = ""
    for row in metrics["rows"]:
        if row["section"] != current_section:
            table_lines.append(f"\n  [{row['section']}]")
            current_section = row["section"]
        winner_marker = ""
        if   row["winner"] == "a":   winner_marker = f" → {sym_a} wins"
        elif row["winner"] == "b":   winner_marker = f" → {sym_b} wins"
        elif row["winner"] == "tie": winner_marker = " → tie"
        table_lines.append(
            f"    {row['metric']}: "
            f"{sym_a}={row['a_display']} | {sym_b}={row['b_display']}"
            f"{winner_marker}"
        )
    table_block = "\n".join(table_lines)

    def fmt_news(news):
        if not news: return "    (no headlines)"
        return "\n".join(
            f"    - [{n['age_days']}d ago] {n['title']}" for n in news[:3]
        )

    def options_snippet(opts):
        if opts is None:
            return "    (not an F&O stock or data unavailable)"
        return (f"    PCR(OI)={opts.pcr_oi:.2f} ({opts.pcr_signal}), "
                f"Max Pain=₹{opts.max_pain:.0f}, "
                f"Composite={opts.composite_signal} {opts.composite_strength}/5, "
                f"Buildup={opts.buildup_signal}")

    def analyst_snippet(ratings, price):
        if ratings.get("data_source") == "unavailable":
            return "    (no analyst data)"
        consensus = ratings.get("consensus", "—")
        pt = ratings.get("price_target") or {}
        target = pt.get("mean")
        if target and price:
            upside = (target - price) / price * 100
            return f"    Consensus={consensus}, Target=₹{target:,.0f} ({upside:+.1f}%)"
        return f"    Consensus={consensus}"

    s = metrics["scores"]
    score_summary = (f"{sym_a} won {s['a']} metrics, {sym_b} won {s['b']}, "
                     f"{s['ties']} tied, {s['na']} not applicable")

    return f"""You are a sell-side equity analyst comparing TWO Indian (NSE) stocks for an Indian retail trader deciding which one to buy. Both stocks may be reasonable holdings; your job is to identify the stronger near-term opportunity (1-4 weeks swing horizon).

STOCK A: {sym_a}
  Current price: ₹{pa.get('current_price', 0):,.2f} ({pa.get('pct_change_1d', 0):+.2f}% today)
  Position: {pa.get('pct_from_52w_high', 0):+.1f}% from 52w high, {pa.get('pct_from_52w_low', 0):+.1f}% from 52w low
  Returns: 30d {pa.get('pct_change_30d', 0):+.2f}%, 90d {pa.get('pct_change_90d', 0):+.2f}%
  RSI: {pa.get('rsi_14', 0)} | MACD hist: {pa.get('macd_hist', 0):+.3f} | Volume: {pa.get('volume_ratio', 0)}x avg
  Options:
{options_snippet(oa)}
  Analyst:
{analyst_snippet(ra, pa.get('current_price'))}
  Recent news:
{fmt_news(a.get('news'))}

STOCK B: {sym_b}
  Current price: ₹{pb.get('current_price', 0):,.2f} ({pb.get('pct_change_1d', 0):+.2f}% today)
  Position: {pb.get('pct_from_52w_high', 0):+.1f}% from 52w high, {pb.get('pct_from_52w_low', 0):+.1f}% from 52w low
  Returns: 30d {pb.get('pct_change_30d', 0):+.2f}%, 90d {pb.get('pct_change_90d', 0):+.2f}%
  RSI: {pb.get('rsi_14', 0)} | MACD hist: {pb.get('macd_hist', 0):+.3f} | Volume: {pb.get('volume_ratio', 0)}x avg
  Options:
{options_snippet(ob)}
  Analyst:
{analyst_snippet(rb, pb.get('current_price'))}
  Recent news:
{fmt_news(b.get('news'))}

METRIC-BY-METRIC COMPARISON
{table_block}

SCORE SUMMARY: {score_summary}

INSTRUCTIONS
Write your verdict in this exact Markdown structure:

## Executive Summary
2-3 sentences capturing the core difference between the two setups and which one looks more compelling right now.

## The Case for {sym_a}
3-4 bullets covering the strongest reasons to pick {sym_a}. Be specific with the numbers.

## The Case for {sym_b}
3-4 bullets covering the strongest reasons to pick {sym_b}. Be specific with the numbers.

## Risk Factors
- For {sym_a}: 1-2 specific risks
- For {sym_b}: 1-2 specific risks

## Verdict

**BETTER BUY**: {sym_a} OR {sym_b} (pick one — be decisive)

OR if neither is compelling right now:

**BETTER AVOID**: both — wait for better setup

Then provide:
- Confidence: Low / Medium / High
- Suggested time horizon
- If allocating capital to BOTH: suggested split (e.g. "70% / 30%") with reasoning
- Or rationale for going 100% on one

## Caveats
2-3 bullets noting what could invalidate this view.

GUARDRAILS
- Don't be wishy-washy. Pick a winner unless both are genuinely poor.
- Use specific numbers from the data above. Don't speak in generalities.
- When two metrics conflict, explicitly say which one you weight more and why.
- End with: "*Comparative technical analysis only. Not investment advice.*"
"""


@app.route("/api/compare", methods=["POST"])
def compare_stocks():
    """
    Compare 2 NSE stocks side-by-side.
    Request body: {"symbols": ["RELIANCE", "HDFCBANK"]}
    """
    if not anthropic_client:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    data = request.get_json(silent=True) or {}
    symbols = data.get("symbols") or []
    if not isinstance(symbols, list) or len(symbols) != 2:
        return jsonify({"error": "Provide exactly 2 symbols in 'symbols' array"}), 400

    symbols = [s.strip().upper() for s in symbols if s]
    if len(symbols) != 2:
        return jsonify({"error": "Empty symbols not allowed"}), 400
    if symbols[0] == symbols[1]:
        return jsonify({"error": "Cannot compare a stock to itself"}), 400

    for sym in symbols:
        if not all(c.isalnum() or c in "&-" for c in sym):
            return jsonify({"error": f"Invalid symbol format: {sym}"}), 400

    logger.info("Comparing %s vs %s", symbols[0], symbols[1])

    # Fetch both bundles in parallel — each bundle itself fans out
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_fetch_stock_bundle, symbols[0])
        future_b = executor.submit(_fetch_stock_bundle, symbols[1])
        bundle_a = future_a.result()
        bundle_b = future_b.result()

    if bundle_a["price"] is None:
        return jsonify({"error": f"Could not fetch data for {symbols[0]}"}), 404
    if bundle_b["price"] is None:
        return jsonify({"error": f"Could not fetch data for {symbols[1]}"}), 404

    metrics = _compute_comparison_metrics(bundle_a, bundle_b)

    prompt = _build_comparison_prompt(bundle_a, bundle_b, metrics)
    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
    except Exception as e:
        logger.exception("Comparison AI call failed: %s", e)
        return jsonify({"error": f"AI comparison failed: {str(e)}"}), 500

    # Convert OptionChainData dataclass → dict for JSON serialization
    def serialize_bundle(bundle):
        b = dict(bundle)
        if b.get("options") is not None:
            o = b["options"]
            b["options"] = {
                "expiry_date":             o.expiry_date,
                "pcr_oi":                  round(o.pcr_oi, 2),
                "pcr_volume":              round(o.pcr_volume, 2),
                "max_pain":                round(o.max_pain, 2),
                "pct_from_max_pain":       round((o.underlying_price - o.max_pain) / o.max_pain * 100, 2) if o.max_pain else 0,
                "highest_call_oi_strike":  o.highest_call_oi_strike,
                "highest_put_oi_strike":   o.highest_put_oi_strike,
                "total_call_oi":           o.total_call_oi,
                "total_put_oi":            o.total_put_oi,
                "pcr_signal":              o.pcr_signal,
                "buildup_signal":          o.buildup_signal,
                "composite_signal":        o.composite_signal,
                "composite_strength":      o.composite_strength,
                "confirmations":           o.confirmations,
            }
        return b

    return jsonify({
        "symbols":     symbols,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "stock_a":     serialize_bundle(bundle_a),
        "stock_b":     serialize_bundle(bundle_b),
        "metrics":     metrics,
        "verdict":     verdict_text,
        "model":       ANTHROPIC_MODEL,
    }), 200


# ===========================================================================
# OPTIONS STRATEGY FEATURE — AI-recommended strategy with concrete legs + P/L
# ===========================================================================

# NSE F&O lot sizes (verify periodically at nseindia.com)
NSE_LOT_SIZES = {
    "RELIANCE": 250, "HDFCBANK": 550, "ICICIBANK": 700, "INFY": 400,
    "TCS": 175, "SBIN": 1500, "BHARTIARTL": 475, "AXISBANK": 625,
    "KOTAKBANK": 400, "LT": 300, "ITC": 1600, "HINDUNILVR": 300,
    "BAJFINANCE": 125, "MARUTI": 50, "ASIANPAINT": 200, "HCLTECH": 350,
    "WIPRO": 3000, "ULTRACEMCO": 100, "SUNPHARMA": 700, "NTPC": 1500,
    "POWERGRID": 1900, "TITAN": 375, "M&M": 350, "TATAMOTORS": 1425,
    "TATASTEEL": 5500, "ADANIENT": 300, "ADANIPORTS": 625, "BAJAJFINSV": 500,
    "NESTLEIND": 250, "ONGC": 4850, "COALINDIA": 2700, "JSWSTEEL": 1350,
    "GRASIM": 475, "INDUSINDBK": 900, "EICHERMOT": 175, "HEROMOTOCO": 300,
    "BAJAJ-AUTO": 125, "BPCL": 1800, "CIPLA": 650, "DIVISLAB": 300,
    "DRREDDY": 125, "BRITANNIA": 200, "TECHM": 600, "APOLLOHOSP": 125,
    "TATACONSUM": 900, "HINDALCO": 2150, "VEDL": 3500, "UPL": 1300,
    "GAIL": 6100, "IOC": 9750,
}


def _get_lot_size(symbol: str) -> int:
    """F&O lot size; default 1 if unknown (caller may want to warn)."""
    return NSE_LOT_SIZES.get(symbol.upper(), 1)


def _build_strategy_prompt(symbol, price_data, options_data, ratings,
                            ctx, outlook, top_pick, alternatives, top_8) -> str:
    """Build Claude prompt asking for strategy explanation + execution plan."""

    def fmt_legs(strat):
        return "\n".join(
            f"    {l.action} {l.quantity} {l.instrument}" +
            (f" @ \u20b9{l.strike:.0f}" if l.strike else "") +
            f" (premium \u20b9{l.premium:.2f})"
            for l in strat.legs
        )

    def fmt_strat(strat):
        max_p = f"\u20b9{strat.max_profit:,.0f}" if strat.max_profit is not None else "Unlimited"
        max_l = f"\u20b9{strat.max_loss:,.0f}"   if strat.max_loss   is not None else "Unlimited"
        be_str = " or ".join(f"\u20b9{b:.2f}" for b in strat.breakevens) if strat.breakevens else "N/A"
        debit_label = "Net Debit" if strat.net_debit > 0 else "Net Credit"
        debit_val = f"\u20b9{abs(strat.net_debit):,.0f}"
        return (
            f"  Strategy: {strat.name} (fit_score={strat.fit_score}/100)\n"
            f"  Category: {strat.category}, Risk: {strat.risk_profile}, Bias: {strat.direction_bias}\n"
            f"  Legs:\n{fmt_legs(strat)}\n"
            f"  {debit_label}: {debit_val}\n"
            f"  Max Profit: {max_p}\n"
            f"  Max Loss: {max_l}\n"
            f"  Breakeven(s): {be_str}\n"
            f"  Capital required: \u20b9{strat.capital_required:,.0f}\n"
            f"  Fit reason: {strat.fit_reason}"
        )

    alts_block = "\n\n".join(
        f"ALTERNATIVE #{i+2}:\n{fmt_strat(a)}" for i, a in enumerate(alternatives)
    ) if alternatives else "(no alternatives)"

    top8_list = "\n".join(
        f"  {i+1}. {r.name} (fit={r.fit_score}, bias={r.direction_bias})"
        for i, r in enumerate(top_8)
    )

    opt_summary = (
        f"PCR(OI)={options_data.pcr_oi:.2f} ({options_data.pcr_signal}), "
        f"Max Pain=\u20b9{options_data.max_pain:.0f}, "
        f"Composite={options_data.composite_signal} ({options_data.composite_strength}/5)"
    )

    pt = (ratings.get("price_target") or {}).get("mean")
    analyst_summary = f"{ratings.get('consensus', 'No data')}"
    if pt:
        upside = (pt - ctx.spot_price) / ctx.spot_price * 100
        analyst_summary += f", Target \u20b9{pt:.0f} ({upside:+.1f}%)"

    return f"""You are an options trading advisor for a retail Indian (NSE) F&O trader. Based on the analysis below, explain the recommended options strategy.

SYMBOL: {symbol}
Spot Price: \u20b9{ctx.spot_price:,.2f}
Lot Size: {ctx.lot_size}
Days to Expiry: {ctx.days_to_expiry}
Expiry Date: {options_data.expiry_date}

MARKET OUTLOOK (derived from technicals + options + analyst)
  Direction: {outlook['direction'].upper()} (bull_signals={outlook['bullish_signals']}, bear_signals={outlook['bearish_signals']})
  Conviction: {outlook['conviction'].upper()}
  IV Regime: {outlook['iv_regime'].upper()}

TECHNICAL CONTEXT
  Price vs SMA: 50={ctx.spot_price/(price_data.get('sma_50') or 1):.2f}x, 200={ctx.spot_price/(price_data.get('sma_200') or 1):.2f}x
  RSI: {price_data.get('rsi_14', 50)} | MACD hist: {price_data.get('macd_hist', 0):+.3f}
  Volume: {price_data.get('volume_ratio', 1):.2f}x avg
  Returns: 30d {price_data.get('pct_change_30d', 0):+.2f}%, 90d {price_data.get('pct_change_90d', 0):+.2f}%

OPTIONS POSITIONING
  {opt_summary}

ANALYST VIEW
  {analyst_summary}

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
RECOMMENDED STRATEGY (highest fit score)
{fmt_strat(top_pick)}

{alts_block}
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

ALL STRATEGIES RANKED BY FIT (top 8):
{top8_list}

INSTRUCTIONS
Write your analysis in this exact Markdown structure:

## Why This Strategy

3-4 sentences explaining why **{top_pick.name}** fits this setup. Reference the actual data — direction, conviction, IV regime — not generic descriptions.

## How to Execute

- **Entry**: When to enter (now vs wait for trigger)
- **Strike rationale**: Why these specific strikes
- **Position sizing**: Standard 1 lot or scaled
- **Profit target**: Specific exit level (% of max or specific price)
- **Stop / Adjustment**: When to cut or roll

## Key Risks

- 2-3 specific risks. Reference current numbers — IV crush, theta decay, gap risk, etc.

## When to Consider Alternatives

- **{alternatives[0].name if alternatives else 'N/A'}**: 1-line condition for when this is better
- **{alternatives[1].name if len(alternatives) > 1 else 'N/A'}**: 1-line condition

## Important Caveats

- Option prices are last-traded, not live bid/ask. Real fills may differ 5-15%.
- Greeks not computed; verify delta/theta/vega in your broker terminal.

GUARDRAILS
- Be concrete with prices and percentages.
- If top fit_score < 50, explicitly note "no high-conviction setup right now."
- End with: "*Strategy analysis based on snapshot data. Not investment advice. Verify live bid/ask before placing orders.*"
"""


@app.route("/api/strategy", methods=["POST"])
def get_strategy():
    """
    Recommend options strategy for a symbol.
    Request body: {"symbol": "RELIANCE"}
    """
    if not anthropic_client:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    data = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    if not all(c.isalnum() or c in "&-" for c in symbol):
        return jsonify({"error": "invalid symbol"}), 400

    logger.info("Strategy analysis for %s", symbol)

    # Fetch data
    price_data = _fetch_price_data(symbol)
    if price_data is None:
        return jsonify({"error": f"Could not fetch price data for {symbol}"}), 404

    spot = price_data["current_price"]
    options_data = fetch_options_data(symbol, spot)
    if options_data is None:
        return jsonify({
            "error": "no_fo_data",
            "message": f"{symbol} is not in F&O segment or option chain unavailable. "
                       "Strategy recommendations require F&O. Try RELIANCE, HDFCBANK, etc."
        }), 404

    ratings = _fetch_analyst_ratings(symbol)

    # Convert option chain to StrategyContext format.
    # options_analyzer now includes call_ltp + put_ltp directly in top_strikes
    # (sourced from Upstox's market_data.ltp, falling back to close_price after
    # market hours). No second HTTP call needed.
    strikes = []
    for s in (options_data.top_strikes or []):
        strikes.append(OptionStrike(
            strike=s["strike"],
            call_ltp=float(s.get("call_ltp") or 0.0),
            call_oi=s.get("call_oi", 0),
            put_ltp=float(s.get("put_ltp") or 0.0),
            put_oi=s.get("put_oi", 0),
            distance_from_spot=s.get("distance_from_spot", 0.0),
        ))

    strikes.sort(key=lambda s: s.strike)

    if len(strikes) < 5:
        return jsonify({
            "error": "insufficient_strikes",
            "message": f"Only {len(strikes)} strikes available — need 5+ for strategy analysis."
        }), 404

    # Final sanity check — at least some strikes must have non-zero LTPs
    strikes_with_premium = sum(1 for s in strikes if s.call_ltp > 0 or s.put_ltp > 0)
    if strikes_with_premium < 3:
        return jsonify({
            "error": "missing_premiums",
            "message": (
                f"Option premiums (LTPs) not available in chain data. "
                f"Only {strikes_with_premium} of {len(strikes)} strikes have premiums. "
                f"Check Render logs for the 'Options chain parsed' line — if it shows "
                f"'0 with LTPs', the Upstox option chain endpoint isn't returning ltp/close_price "
                f"for this symbol (possibly illiquid expiry). Try a more active F&O stock."
            )
        }), 500

    outlook = derive_outlook(price_data, options_data, ratings)

    try:
        from datetime import datetime as dt
        expiry_dt = dt.strptime(options_data.expiry_date, "%Y-%m-%d")
        days_to_expiry = max((expiry_dt.date() - dt.now().date()).days, 1)
    except Exception:
        days_to_expiry = 7

    target_price = (ratings.get("price_target") or {}).get("mean") or (spot * 1.05)
    stop_price = price_data.get("sma_50") or (spot * 0.95)

    ctx = StrategyContext(
        symbol=symbol,
        spot_price=spot,
        lot_size=_get_lot_size(symbol),
        strikes=strikes,
        direction=outlook["direction"],
        conviction=outlook["conviction"],
        iv_regime=outlook["iv_regime"],
        target_price=target_price,
        stop_price=stop_price,
        atr_14=price_data.get("atr_14", spot * 0.02),
        days_to_expiry=days_to_expiry,
    )

    all_results = build_all_strategies(ctx)
    if not all_results:
        return jsonify({"error": "No applicable strategies for current chain data"}), 500

    top_pick = all_results[0]
    alternatives = all_results[1:3]

    prompt = _build_strategy_prompt(symbol, price_data, options_data, ratings,
                                     ctx, outlook, top_pick, alternatives,
                                     all_results[:8])

    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = "".join(b.text for b in response.content if hasattr(b, "text"))
    except Exception as e:
        logger.exception("Strategy AI call failed: %s", e)
        return jsonify({"error": f"AI strategy analysis failed: {str(e)}"}), 500

    return jsonify({
        "symbol": symbol,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "spot_price": spot,
        "lot_size": ctx.lot_size,
        "days_to_expiry": days_to_expiry,
        "expiry_date": options_data.expiry_date,
        "outlook": outlook,
        "top_pick": top_pick.to_dict(),
        "alternatives": [a.to_dict() for a in alternatives],
        "all_results_summary": [
            {
                "name": r.name,
                "fit_score": r.fit_score,
                "direction_bias": r.direction_bias,
                "risk_profile": r.risk_profile,
            }
            for r in all_results
        ],
        "ai_analysis": ai_text,
        "model": ANTHROPIC_MODEL,
    }), 200


# ===========================================================================
# POSITION INSIGHTS — AI-powered guidance for an existing trade
# ===========================================================================
#
# Endpoint: POST /api/position
# Body: { symbol, entry_price, direction ("long"|"short"), quantity }
#
# Reuses the same data-gathering as /api/analyze (price, options, ratings,
# news) but reframes the AI prompt to be position-aware instead of generic.
# Computes position-specific metrics (unrealized P&L, distance to key option
# strikes, etc.) and asks Claude for hold/exit/scale guidance.
# ===========================================================================

def _compute_position_metrics(
    entry_price: float,
    direction: str,
    quantity: int,
    current_price: float,
    options_data: Optional[OptionChainData],
) -> dict:
    """
    Compute position-specific metrics.

    Direction-aware: for shorts, P&L is reversed.
    """
    # Direction multiplier: +1 for long, -1 for short
    dir_mult = 1 if direction == "long" else -1

    price_move      = current_price - entry_price
    pnl_per_unit    = price_move * dir_mult
    unrealized_pnl  = pnl_per_unit * quantity
    pnl_pct         = (pnl_per_unit / entry_price) * 100 if entry_price > 0 else 0
    invested        = entry_price * quantity

    # Where is the position relative to key option-derived levels?
    nearest_resistance = None
    nearest_support    = None
    distance_to_max_pain = None
    if options_data:
        # Use highest OI strikes as default S/R
        nearest_resistance = options_data.highest_call_oi_strike or None
        nearest_support    = options_data.highest_put_oi_strike or None
        if options_data.max_pain:
            distance_to_max_pain = (
                (current_price - options_data.max_pain) / options_data.max_pain * 100
            )

    return {
        "entry_price":         entry_price,
        "direction":           direction,
        "quantity":            quantity,
        "current_price":       current_price,
        "invested_amount":     round(invested, 2),
        "current_value":       round(current_price * quantity, 2),
        "unrealized_pnl":      round(unrealized_pnl, 2),
        "unrealized_pnl_pct":  round(pnl_pct, 2),
        "price_move_abs":      round(price_move, 2),
        "price_move_pct":      round((price_move / entry_price * 100) if entry_price > 0 else 0, 2),
        "nearest_resistance":  nearest_resistance,
        "nearest_support":     nearest_support,
        "distance_to_max_pain": round(distance_to_max_pain, 2) if distance_to_max_pain is not None else None,
    }


def _build_position_prompt(
    symbol: str,
    pos_metrics: dict,
    price_data: dict,
    options_data: Optional[OptionChainData],
    ratings: Optional[dict],
    news: list,
) -> str:
    """
    Build a position-aware prompt for Claude.

    Mental frame: "I'm holding this position right now, what should I do?"
    Not "Should I open this position?" — those are different questions.
    """
    p           = price_data
    direction   = pos_metrics["direction"]
    entry       = pos_metrics["entry_price"]
    qty         = pos_metrics["quantity"]
    cur         = pos_metrics["current_price"]
    pnl         = pos_metrics["unrealized_pnl"]
    pnl_pct     = pos_metrics["unrealized_pnl_pct"]
    invested    = pos_metrics["invested_amount"]
    cur_val     = pos_metrics["current_value"]

    pnl_status = "IN PROFIT" if pnl > 0 else ("IN LOSS" if pnl < 0 else "FLAT")

    # ----- Section: Position Snapshot -----
    position_block = (
        f"## CURRENT POSITION\n"
        f"- Symbol:           {symbol}\n"
        f"- Direction:        {direction.upper()}\n"
        f"- Quantity:         {qty} shares\n"
        f"- Entry price:      ₹{entry:.2f}\n"
        f"- Current price:    ₹{cur:.2f}\n"
        f"- Price move:       {pos_metrics['price_move_abs']:+.2f} ({pos_metrics['price_move_pct']:+.2f}%)\n"
        f"- Invested amount:  ₹{invested:,.2f}\n"
        f"- Current value:    ₹{cur_val:,.2f}\n"
        f"- Unrealized P&L:   ₹{pnl:,.2f} ({pnl_pct:+.2f}%) — {pnl_status}\n"
    )

    # ----- Section: Technicals -----
    tech_block = (
        f"\n## TECHNICAL CONTEXT\n"
        f"- RSI(14):     {p['rsi_14']}\n"
        f"- ATR(14):     ₹{p.get('atr_14', 0):.2f} (avg daily range)\n"
        f"- SMA 20:      ₹{p.get('sma_20', 0):.2f}\n"
        f"- SMA 50:      ₹{p.get('sma_50', 0):.2f}\n"
        f"- SMA 200:     ₹{p.get('sma_200', 0):.2f}\n"
        f"- BB Upper:    ₹{p.get('bb_upper', 0):.2f}\n"
        f"- BB Lower:    ₹{p.get('bb_lower', 0):.2f}\n"
        f"- MACD:        {p.get('macd', 0):.2f} (signal {p.get('macd_signal', 0):.2f})\n"
        f"- 1D change:   {p['pct_change_1d']:+.2f}%\n"
        f"- 30D change:  {p['pct_change_30d']:+.2f}%\n"
        f"- 52W range:   ₹{p['low_52w']:.2f} – ₹{p['high_52w']:.2f}\n"
        f"- Volume:      {p['volume_ratio']:.1f}x avg\n"
    )

    # ----- Section: Options Context -----
    options_block = ""
    if options_data:
        nearest_r = pos_metrics["nearest_resistance"]
        nearest_s = pos_metrics["nearest_support"]
        options_block = (
            f"\n## OPTIONS SENTIMENT (F&O expiry {options_data.expiry_date})\n"
            f"- PCR (OI):              {options_data.pcr_oi:.2f} → {options_data.pcr_signal}\n"
            f"- Max Pain:              ₹{options_data.max_pain:.2f} "
            f"(spot is {pos_metrics['distance_to_max_pain']:+.2f}% from it)\n"
            f"- OI Buildup:            {options_data.buildup_signal}\n"
            f"- Composite signal:      {options_data.composite_signal} "
            f"({options_data.composite_strength}/5)\n"
            f"- Highest Call OI (resistance): ₹{nearest_r}\n"
            f"- Highest Put OI (support):     ₹{nearest_s}\n"
        )
        if options_data.confirmations:
            confs = "\n".join(f"  • {c}" for c in options_data.confirmations[:5])
            options_block += f"- Signal rationale:\n{confs}\n"
    else:
        options_block = "\n## OPTIONS SENTIMENT\nNot available (non-F&O stock).\n"

    # ----- Section: Analyst Ratings -----
    analyst_block = ""
    if ratings and ratings.get("data_source") != "unavailable":
        pt = ratings.get("price_target") or {}
        consensus = ratings.get("consensus") or "Unknown"
        target_mean = pt.get("mean")
        analyst_block = f"\n## ANALYST CONSENSUS\n- Rating: {consensus}\n"
        if target_mean:
            upside = ((target_mean - cur) / cur) * 100
            analyst_block += (
                f"- Mean target: ₹{target_mean:.2f} ({upside:+.2f}% from current)\n"
            )
        if pt.get("low") and pt.get("high"):
            analyst_block += f"- Target range: ₹{pt['low']:.2f} – ₹{pt['high']:.2f}\n"

    # ----- Section: News -----
    news_block = ""
    if news:
        news_block = "\n## RECENT NEWS (last 7 days)\n"
        for item in news[:5]:
            age = "today" if item.get("age_days") == 0 else f"{item.get('age_days')}d ago"
            news_block += f"- [{age}] {item.get('title', '')[:120]}\n"

    # ----- Bring it all together -----
    return f"""You are a senior portfolio manager advising a retail F&O trader on the
Indian NSE about an EXISTING position they already hold. The trader is NOT
asking whether to open this trade — they're already in. Your job is to help
them MANAGE it: hold, exit, scale, or adjust risk.

{position_block}
{tech_block}
{options_block}
{analyst_block}
{news_block}

## YOUR TASK

Write a focused, actionable analysis in markdown with these EXACT sections:

### 1. Verdict
A one-line directional call: **STRONG HOLD** / **HOLD** / **TRIM** / **EXIT** / **CONSIDER SCALING UP** / **CONSIDER SCALING DOWN**.
Follow with 2-3 sentences justifying the verdict using the strongest signals above.

### 2. Critical Price Points
Concrete levels the trader must watch, with brief rationale:
- **Profit lock zone:** where to consider booking partial profits (cite the technical or option level)
- **Stop loss suggestion:** based on ATR / BBands / OI support, NOT entry price
- **Breakout/breakdown trigger:** the level that would invalidate or strengthen the thesis
- **Target zone:** where to fully exit if the trade works

For a {direction.upper()} position, "profit zone" is above entry for long / below entry for short.

### 3. Exit Strategy
A step-by-step plan:
- When to take partial profits (% of position to trim, at what price)
- When to move stop to breakeven
- When to fully exit
- Time-based exit: if the position goes nowhere for N days, what's the rule?

### 4. Risk Management Adjustments
- Recommended trailing stop methodology (e.g., "2× ATR below 20DMA")
- Position size review: is the current ₹{cur_val:,.0f} exposure appropriate given the volatility shown?
- Hedge consideration: would a protective option leg make sense here?

### 5. Scaling Decision
Given the current setup, should the trader:
- **Add more** (and what conditions would trigger that)
- **Reduce** (and how much)
- **Hold size as-is**
Explain the reasoning.

### 6. Time Horizon & Watch List
- Expected holding duration for the thesis to play out (days/weeks/months)
- Specific upcoming events to monitor (earnings, expiry, sector news)
- A clear "exit signal checklist" — bullet 3-4 conditions that would force an exit regardless of price

### 7. Important Caveats
- Acknowledge what's unknown (no stop loss info from user, etc.)
- Note any specific risks given the {direction.upper()} direction

Be direct and concrete. Use actual ₹ values, not vague language. The trader needs
decisions they can act on TODAY, not generic advice."""


@app.route("/api/position", methods=["POST"])
def get_position_insights():
    """
    AI-powered insights on an existing position.

    Body:
      {
        "symbol":      "RELIANCE",
        "entry_price": 2450.00,
        "direction":   "long" | "short",
        "quantity":    50
      }

    Returns JSON with position metrics + AI guidance.
    """
    if not anthropic_client:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    body = request.get_json(silent=True) or {}

    # --- Input validation ---
    symbol = (body.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    if not all(c.isalnum() or c in "&-" for c in symbol):
        return jsonify({"error": "invalid symbol"}), 400

    try:
        entry_price = float(body.get("entry_price", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "entry_price must be a number"}), 400
    if entry_price <= 0:
        return jsonify({"error": "entry_price must be > 0"}), 400

    direction = (body.get("direction") or "long").strip().lower()
    if direction not in ("long", "short"):
        return jsonify({"error": "direction must be 'long' or 'short'"}), 400

    try:
        quantity = int(body.get("quantity", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "quantity must be an integer"}), 400
    if quantity <= 0:
        return jsonify({"error": "quantity must be > 0"}), 400

    logger.info("Position insights for %s (%s %d @ ₹%.2f)",
                symbol, direction, quantity, entry_price)

    # --- Fetch data in parallel (same pattern as /api/compare) ---
    price_data = _fetch_price_data(symbol)
    if price_data is None:
        return jsonify({"error": f"Could not fetch data for {symbol}"}), 404

    current_price = price_data["current_price"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_ratings = executor.submit(_fetch_analyst_ratings, symbol)
        future_options = executor.submit(fetch_options_data, symbol, current_price)
        future_news    = executor.submit(_fetch_news, symbol, 5)

        ratings      = future_ratings.result()
        options_data = future_options.result()
        news         = future_news.result()

    # --- Compute position-specific metrics ---
    pos_metrics = _compute_position_metrics(
        entry_price, direction, quantity, current_price, options_data
    )

    # --- AI analysis ---
    prompt = _build_position_prompt(
        symbol, pos_metrics, price_data, options_data, ratings, news
    )

    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
    except Exception as e:
        logger.exception("Anthropic API call failed: %s", e)
        return jsonify({"error": f"AI analysis failed: {str(e)}"}), 500

    # --- Serialize options data for the dashboard (same shape as /api/analyze) ---
    options_payload = None
    if options_data is not None:
        options_payload = {
            "expiry_date":       options_data.expiry_date,
            "pcr_oi":            round(options_data.pcr_oi, 2),
            "max_pain":          round(options_data.max_pain, 2),
            "pct_from_max_pain": round((options_data.underlying_price - options_data.max_pain) / options_data.max_pain * 100, 2) if options_data.max_pain else 0,
            "highest_call_oi_strike": options_data.highest_call_oi_strike,
            "highest_put_oi_strike":  options_data.highest_put_oi_strike,
            "pcr_signal":             options_data.pcr_signal,
            "buildup_signal":         options_data.buildup_signal,
            "composite_signal":       options_data.composite_signal,
            "composite_strength":     options_data.composite_strength,
            "confirmations":          options_data.confirmations,
        }

    return jsonify({
        "symbol":         symbol,
        "analyzed_at":    datetime.utcnow().isoformat() + "Z",
        "position":       pos_metrics,
        "price_data":     price_data,
        "options":        options_payload,
        "ratings":        ratings,
        "news":           news,
        "ai_analysis":    ai_text,
        "model":          ANTHROPIC_MODEL,
    }), 200


# ===========================================================================
# TRADE PLAN — algorithmic levels + AI commentary for a new entry
# ===========================================================================
#
# Endpoint: POST /api/trade-plan
# Body:
#   { symbol, direction ("long"|"short"|"hedged"),
#     input_mode ("qty"|"capital"), qty?, capital? }
#
# Returns a complete trade plan: entry zone, stop loss, position add zones,
# laddered profit targets, risk metrics, optional hedge leg (if hedged),
# AI commentary explaining the rationale.
#
# Design:
#   - Reuse /api/analyze data-fetching (price + options + ratings + news)
#   - Compute levels algorithmically from objective signals (ATR, OI strikes,
#     SMAs, 52w high/low) — no AI needed for these
#   - AI writes the prose explaining WHY each level was chosen and what
#     conditions would invalidate the plan
# ===========================================================================

def _compute_trade_levels(
    direction: str,           # "long" | "short" | "hedged"
    price_data: dict,
    options_data: Optional[OptionChainData],
    ratings: Optional[dict],
) -> dict:
    """
    Algorithmically compute entry zone, stop, add zones, and laddered targets.

    Returns a dict with all numeric levels. AI prompt later uses these to
    write the commentary; frontend renders them in cards.

    All prices are direction-aware:
      - For LONG: stops below entry, targets above
      - For SHORT: stops above entry, targets below
      - For HEDGED: same as LONG but with a protective put leg
    """
    price   = price_data["current_price"]
    atr     = price_data.get("atr_14") or (price * 0.02)  # 2% fallback
    sma_20  = price_data.get("sma_20")  or price
    sma_50  = price_data.get("sma_50")  or price
    sma_200 = price_data.get("sma_200") or price
    bb_upper = price_data.get("bb_upper") or (price + 2 * atr)
    bb_lower = price_data.get("bb_lower") or (price - 2 * atr)
    high_52w = price_data.get("high_52w") or price
    low_52w  = price_data.get("low_52w")  or price

    # Option-derived levels (high OI strikes act as dynamic S/R)
    nearest_resistance = None  # highest Call OI strike (where sellers cluster)
    nearest_support    = None  # highest Put OI strike (where buyers defend)
    if options_data:
        nearest_resistance = options_data.highest_call_oi_strike or None
        nearest_support    = options_data.highest_put_oi_strike or None

    # Analyst-derived levels
    analyst_target = None
    if ratings and ratings.get("price_target"):
        analyst_target = ratings["price_target"].get("mean")

    # For HEDGED, we treat the underlying entry the same as LONG;
    # the protective put is computed separately below.
    is_long = direction in ("long", "hedged")
    dir_mult = 1 if is_long else -1

    # ----- Entry zone (a range, not a single price) -----
    # LONG: prefer entering on a small pullback toward (current - 0.5*ATR)
    # SHORT: prefer entering on a small pop toward (current + 0.5*ATR)
    if is_long:
        entry_zone_high = price                           # market entry top
        entry_zone_low  = price - 0.5 * atr               # patient entry
    else:
        entry_zone_low  = price                           # market entry bottom
        entry_zone_high = price + 0.5 * atr               # patient entry
    entry_mid = (entry_zone_low + entry_zone_high) / 2

    # ----- Stop loss — multi-candidate, pick the tightest reasonable -----
    # LONG: stop must be BELOW entry. Pick the highest of valid candidates
    #   (highest = tightest for long).
    # SHORT: stop must be ABOVE entry. Pick the lowest valid candidate.
    stop_candidates = []
    if is_long:
        candidates = [
            {"value": entry_mid - 2 * atr,        "source": f"Entry − 2×ATR (₹{atr:.2f})"},
            {"value": sma_50 * 0.99,              "source": "1% below SMA 50"},
            {"value": bb_lower,                   "source": "Bollinger lower band"},
        ]
        if nearest_support:
            # 0.5% below the highest-put-OI strike acts as confirmation level
            candidates.append({
                "value": nearest_support * 0.995,
                "source": f"Just below option support ₹{nearest_support:.0f}",
            })
        valid = [c for c in candidates if 0 < c["value"] < entry_mid]
        if valid:
            chosen = max(valid, key=lambda c: c["value"])  # tightest = highest
        else:
            chosen = {"value": entry_mid * 0.98, "source": "2% below entry (fallback)"}
    else:
        candidates = [
            {"value": entry_mid + 2 * atr,        "source": f"Entry + 2×ATR (₹{atr:.2f})"},
            {"value": sma_50 * 1.01,              "source": "1% above SMA 50"},
            {"value": bb_upper,                   "source": "Bollinger upper band"},
        ]
        if nearest_resistance:
            candidates.append({
                "value": nearest_resistance * 1.005,
                "source": f"Just above option resistance ₹{nearest_resistance:.0f}",
            })
        valid = [c for c in candidates if c["value"] > entry_mid]
        if valid:
            chosen = min(valid, key=lambda c: c["value"])  # tightest = lowest
        else:
            chosen = {"value": entry_mid * 1.02, "source": "2% above entry (fallback)"}

    stop_loss      = chosen["value"]
    stop_source    = chosen["source"]
    risk_per_share = abs(entry_mid - stop_loss)

    # ----- Position add zones -----
    # Two scenarios: averaging on adverse move (dollar-cost) OR pyramiding on confirmation
    # We provide both — user decides which approach to use based on trade thesis.
    if is_long:
        add_adverse_1 = entry_mid * 0.97       # add at -3%
        add_adverse_2 = entry_mid * 0.95       # add at -5%
        add_confirm_1 = entry_mid * 1.02       # add at +2% breakout
        add_confirm_2 = entry_mid * 1.04       # add at +4% confirmation
    else:
        add_adverse_1 = entry_mid * 1.03       # add at +3% (adverse for short)
        add_adverse_2 = entry_mid * 1.05       # add at +5%
        add_confirm_1 = entry_mid * 0.98       # add at -2% breakdown
        add_confirm_2 = entry_mid * 0.96       # add at -4% confirmation

    # ----- Laddered profit targets (3 levels, 33/33/34 split) -----
    # T1 = 1.5R from entry (mechanical, locks some profit)
    # T2 = nearest opposing OI strike or analyst target, whichever is closer
    # T3 = 3R or 52w high/low, whichever is closer
    t1 = entry_mid + (1.5 * risk_per_share * dir_mult)

    # T2 candidates
    t2_candidates = []
    if is_long:
        if nearest_resistance and nearest_resistance > entry_mid:
            t2_candidates.append(nearest_resistance)
        if analyst_target and analyst_target > entry_mid:
            t2_candidates.append(analyst_target)
        if high_52w > entry_mid * 1.02:
            t2_candidates.append(high_52w * 0.99)  # just below 52w high
        if not t2_candidates:
            t2_candidates.append(entry_mid + (2.0 * risk_per_share))
        t2 = min(t2_candidates)  # closest target
    else:
        if nearest_support and nearest_support < entry_mid:
            t2_candidates.append(nearest_support)
        if analyst_target and analyst_target < entry_mid:
            t2_candidates.append(analyst_target)
        if low_52w < entry_mid * 0.98:
            t2_candidates.append(low_52w * 1.01)  # just above 52w low
        if not t2_candidates:
            t2_candidates.append(entry_mid - (2.0 * risk_per_share))
        t2 = max(t2_candidates)

    # T3 = stretch target
    t3_mechanical = entry_mid + (3.0 * risk_per_share * dir_mult)
    if is_long:
        t3 = min(t3_mechanical, high_52w * 1.05)  # cap at 5% above 52w high
    else:
        t3 = max(t3_mechanical, low_52w * 0.95)

    return {
        "direction":        direction,
        "current_price":    round(price, 2),
        "atr":              round(atr, 2),
        "entry_zone": {
            "low":    round(entry_zone_low, 2),
            "mid":    round(entry_mid, 2),
            "high":   round(entry_zone_high, 2),
        },
        "stop_loss": {
            "price":           round(stop_loss, 2),
            "source":          stop_source,
            "distance_pct":    round((stop_loss - entry_mid) / entry_mid * 100, 2),
            "risk_per_share":  round(risk_per_share, 2),
        },
        "add_zones": {
            "adverse_1":  round(add_adverse_1, 2),
            "adverse_2":  round(add_adverse_2, 2),
            "confirm_1":  round(add_confirm_1, 2),
            "confirm_2":  round(add_confirm_2, 2),
        },
        "targets": {
            "t1": {
                "price":      round(t1, 2),
                "label":      "Take 33%",
                "rationale":  "1.5R from entry — mechanical first profit lock",
                "r_multiple": 1.5,
                "pct_move":   round((t1 - entry_mid) / entry_mid * 100, 2),
            },
            "t2": {
                "price":      round(t2, 2),
                "label":      "Take 33%",
                "rationale":  _t2_rationale(is_long, t2, nearest_resistance, nearest_support, analyst_target, high_52w, low_52w),
                "r_multiple": round(abs(t2 - entry_mid) / risk_per_share, 2) if risk_per_share > 0 else 0,
                "pct_move":   round((t2 - entry_mid) / entry_mid * 100, 2),
            },
            "t3": {
                "price":      round(t3, 2),
                "label":      "Take 34% (final exit)",
                "rationale":  "Stretch target — 3R or near 52w extreme",
                "r_multiple": round(abs(t3 - entry_mid) / risk_per_share, 2) if risk_per_share > 0 else 0,
                "pct_move":   round((t3 - entry_mid) / entry_mid * 100, 2),
            },
        },
        "reference_levels": {
            "sma_20":              round(sma_20, 2),
            "sma_50":              round(sma_50, 2),
            "sma_200":             round(sma_200, 2),
            "bb_upper":            round(bb_upper, 2),
            "bb_lower":            round(bb_lower, 2),
            "high_52w":            round(high_52w, 2),
            "low_52w":             round(low_52w, 2),
            "nearest_resistance":  nearest_resistance,
            "nearest_support":     nearest_support,
            "analyst_target":      analyst_target,
        },
    }


def _t2_rationale(is_long, t2, nr, ns, at, h52, l52):
    """Generate a human-readable rationale for the T2 target choice."""
    eps = 0.005  # 0.5% tolerance for matching
    if is_long:
        if nr and abs(t2 - nr) / nr < eps:
            return f"Nearest option resistance (highest Call OI ₹{nr:.0f})"
        if at and abs(t2 - at) / at < eps:
            return f"Analyst mean target (₹{at:.0f})"
        if h52 and abs(t2 - h52 * 0.99) / h52 < eps:
            return "Just below 52-week high"
    else:
        if ns and abs(t2 - ns) / ns < eps:
            return f"Nearest option support (highest Put OI ₹{ns:.0f})"
        if at and abs(t2 - at) / at < eps:
            return f"Analyst mean target (₹{at:.0f})"
        if l52 and abs(t2 - l52 * 1.01) / l52 < eps:
            return "Just above 52-week low"
    return "2R from entry — fallback"


def _compute_hedge_leg(
    price: float, qty: int, options_data: Optional[OptionChainData]
) -> Optional[dict]:
    """
    For HEDGED direction, propose a protective put leg.

    Strategy: buy 1 ATM put per lot of stock (or per equivalent qty of shares
    that approximates one lot). The put cost is the "insurance premium" you
    pay to cap downside.
    """
    if not options_data or not options_data.top_strikes:
        return None

    # Find the ATM put (strike closest to current price, with non-zero LTP)
    candidates = []
    for s in options_data.top_strikes:
        ltp = s.get("put_ltp") or 0
        if ltp <= 0:
            continue
        distance = abs(s["strike"] - price)
        candidates.append((distance, s["strike"], ltp))
    if not candidates:
        return None

    candidates.sort()
    _, strike, ltp = candidates[0]

    # How many puts to buy? Depends on lot size of the underlying.
    # For a position of `qty` shares, we need `qty / lot_size` puts (rounded up).
    # But the symbol isn't passed here so we'll just report the per-lot cost
    # and let the frontend multiply by required lots.
    return {
        "leg_type":           "protective_put",
        "strike":             strike,
        "premium_per_share":  round(ltp, 2),
        "cost_per_lot":       None,  # filled by caller who knows the lot size
        "max_loss_per_share": round(price - strike + ltp, 2),
        "expiry":             options_data.expiry_date,
    }


def _compute_position_economics(
    levels: dict, qty: int, lot_size: int, hedge: Optional[dict],
) -> dict:
    """Compute risk in ₹ and reward at each target, given the position size."""
    entry_mid    = levels["entry_zone"]["mid"]
    stop         = levels["stop_loss"]["price"]
    risk_per_shr = levels["stop_loss"]["risk_per_share"]
    direction    = levels["direction"]
    is_long      = direction in ("long", "hedged")

    capital_required = round(entry_mid * qty, 2)
    base_risk = round(risk_per_shr * qty, 2)

    # Adjust risk if hedged: protective put caps downside
    hedge_cost = 0.0
    effective_risk = base_risk
    if direction == "hedged" and hedge:
        # Calculate hedge cost based on lot size — round UP to nearest lot
        # to avoid under-hedging
        lots_needed = max(1, -(-qty // lot_size))   # ceiling division
        hedge_cost  = round(hedge["premium_per_share"] * lot_size * lots_needed, 2)
        hedge["cost_per_lot"]   = round(hedge["premium_per_share"] * lot_size, 2)
        hedge["lots_required"]  = lots_needed
        hedge["total_cost"]     = hedge_cost
        # With protective put at strike K, max loss per share = (entry - K + premium)
        # So total max loss = qty × (entry - K) + hedge_cost
        # Cap this at the algorithmic stop OR the put strike, whichever is closer
        put_strike_loss = (entry_mid - hedge["strike"]) * qty + hedge_cost
        # Compare to vanilla stop loss
        effective_risk = min(base_risk + hedge_cost, max(0, put_strike_loss))
        effective_risk = round(effective_risk, 2)

    # Reward at each target
    targets_econ = {}
    for tier, t in levels["targets"].items():
        target_price = t["price"]
        # 33/33/34 split — what's the reward per slice?
        slice_qty = qty // 3 if tier in ("t1", "t2") else qty - 2 * (qty // 3)
        gross_reward = (target_price - entry_mid) * slice_qty
        if not is_long:
            gross_reward = (entry_mid - target_price) * slice_qty
        targets_econ[tier] = {
            "slice_qty":   slice_qty,
            "exit_value":  round(target_price * slice_qty, 2),
            "gross_reward": round(gross_reward, 2),
        }

    # Total reward if all targets hit
    total_reward = sum(targets_econ[t]["gross_reward"] for t in targets_econ)
    # Minus hedge cost if applicable
    net_reward = round(total_reward - hedge_cost, 2)

    return {
        "qty":                qty,
        "capital_required":   capital_required,
        "base_risk":          base_risk,
        "hedge_cost":         hedge_cost,
        "effective_risk":     effective_risk,
        "max_reward":         net_reward,
        "risk_reward_ratio":  round(net_reward / effective_risk, 2) if effective_risk > 0 else 0,
        "targets_economics":  targets_econ,
    }


def _build_trade_plan_prompt(
    symbol: str, direction: str, levels: dict, econ: dict,
    hedge: Optional[dict], price_data: dict, options_data: Optional[OptionChainData],
    ratings: Optional[dict],
) -> str:
    """
    Build a focused AI prompt — algo already computed the numbers, AI just
    explains the rationale.
    """
    p = price_data
    entry = levels["entry_zone"]
    stop  = levels["stop_loss"]
    tgts  = levels["targets"]

    direction_label = {
        "long":   "LONG (buy)",
        "short":  "SHORT (sell)",
        "hedged": "HEDGED LONG (buy stock + protective put)",
    }[direction]

    hedge_block = ""
    if direction == "hedged" and hedge:
        hedge_block = (
            f"\n## HEDGE LEG\n"
            f"- Protective ATM put strike: ₹{hedge['strike']}\n"
            f"- Put premium: ₹{hedge['premium_per_share']}/share\n"
            f"- Total hedge cost: ₹{hedge['total_cost']:,.0f}\n"
            f"- This caps your max loss at ₹{econ['effective_risk']:,.0f} "
            f"vs ₹{econ['base_risk']:,.0f} unhedged\n"
        )

    options_block = ""
    if options_data:
        options_block = (
            f"\n## OPTIONS CONTEXT (F&O {options_data.expiry_date})\n"
            f"- PCR (OI): {options_data.pcr_oi:.2f} → {options_data.pcr_signal}\n"
            f"- Max Pain: ₹{options_data.max_pain:.2f}\n"
            f"- Composite signal: {options_data.composite_signal}\n"
            f"- Highest Call OI (resistance): ₹{options_data.highest_call_oi_strike}\n"
            f"- Highest Put OI (support): ₹{options_data.highest_put_oi_strike}\n"
        )

    analyst_block = ""
    if ratings and ratings.get("price_target"):
        pt = ratings["price_target"]
        analyst_block = (
            f"\n## ANALYST VIEW\n"
            f"- Consensus: {ratings.get('consensus', '—')}\n"
            f"- Mean target: ₹{pt.get('mean', '—')}\n"
        )

    return f"""You are a senior trading desk strategist writing an execution memo for a
retail Indian F&O trader who has decided to take a {direction_label} position in
{symbol}. The numerical levels below have already been computed algorithmically.
Your job is NOT to second-guess them — they're based on objective signals
(ATR, OI strikes, SMAs, 52w levels). Your job is to write the COMMENTARY that
explains WHY each level makes sense and WHAT TO WATCH FOR.

## MARKET CONTEXT
- Current price: ₹{p['current_price']:.2f}
- ATR(14): ₹{levels['atr']:.2f}
- 30D change: {p['pct_change_1d']:+.2f}% (today) · 1D: {p['pct_change_1d']:+.2f}%
- RSI(14): {p['rsi_14']}
- SMA 50/200: ₹{p.get('sma_50', 0):.2f} / ₹{p.get('sma_200', 0):.2f}
- 52w range: ₹{p['low_52w']:.2f} – ₹{p['high_52w']:.2f}
{options_block}{analyst_block}

## ALGORITHMIC TRADE PLAN (already computed)

### Entry
- Patient entry zone: ₹{entry['low']:.2f} – ₹{entry['high']:.2f}
- Midpoint used for risk calc: ₹{entry['mid']:.2f}

### Stop loss
- Price: ₹{stop['price']:.2f} ({stop['distance_pct']:+.2f}% from entry)
- Rationale: {stop['source']}
- Risk per share: ₹{stop['risk_per_share']:.2f}

### Position adds (averaging / scaling)
- Add on adverse move: ₹{levels['add_zones']['adverse_1']:.2f}, ₹{levels['add_zones']['adverse_2']:.2f}
- Add on confirmation: ₹{levels['add_zones']['confirm_1']:.2f}, ₹{levels['add_zones']['confirm_2']:.2f}

### Profit targets (ladder)
- T1 (33% exit): ₹{tgts['t1']['price']:.2f} ({tgts['t1']['r_multiple']:.1f}R) — {tgts['t1']['rationale']}
- T2 (33% exit): ₹{tgts['t2']['price']:.2f} ({tgts['t2']['r_multiple']:.1f}R) — {tgts['t2']['rationale']}
- T3 (34% exit): ₹{tgts['t3']['price']:.2f} ({tgts['t3']['r_multiple']:.1f}R) — {tgts['t3']['rationale']}

### Position economics
- Quantity: {econ['qty']} shares
- Capital deployed: ₹{econ['capital_required']:,.0f}
- Total risk: ₹{econ['effective_risk']:,.0f}
- Max reward (all targets): ₹{econ['max_reward']:,.0f}
- Risk:Reward ratio: 1:{econ['risk_reward_ratio']:.2f}
{hedge_block}

## YOUR TASK

Write a focused execution memo in markdown with these EXACT sections (no introduction,
no preamble — start straight with the first heading):

### Psychology Check
This is the FIRST section because mental discipline matters more than perfect entries.
Write 2 parts:
1. **Setup-specific psychology (2-3 sentences)**: Identify the SPECIFIC emotional traps
   this {direction_label} setup will trigger for the trader. Examples to draw from:
   - LONG at extended price: "First 1% gain will feel like 'easy money, take it now'. Don't."
   - LONG below SMA 200: "When it bounces 3% and stalls, you'll feel vindicated and add too early."
   - SHORT against bullish consensus: "Every broker upgrade will make you question the thesis."
   - SHORT in strong uptrend: "Squeeze risk is real; one 5% gap-up will test your conviction."
   - HEDGED: "If trade works, the put will feel like wasted money — don't unwind early."
   - Hedged + losing: "Don't roll the put down to 'save money' — that's removing the hedge."
   Tie this directly to today's specific setup (cite the RSI, trend, or option signal).

2. **Universal reminders (2-3 bullets max)**: Brief, not preachy. Pick the most relevant
   from: FOMO management, revenge trading after a stop-out, sunk cost on losing positions,
   over-trading after a winner, journaling the plan vs the execution.

### Entry Reasoning
2-3 sentences on why this entry zone makes sense given today's setup. Reference the
current momentum, RSI, recent price action. If the trade is going WITH the trend
(e.g., LONG on a stock above its rising SMAs), say so. If it's a counter-trend
play, flag the additional risk.

### Stop Loss Logic
2-3 sentences explaining what would invalidate this trade. Reference the specific
source of the stop (e.g., "the SMA 50 has acted as support 3x this quarter").
Be specific about what price action you'd see if the stop gets hit.

### Add Strategy
2-3 sentences on when to use adverse-move adds vs confirmation adds. Explicitly
recommend ONE approach as default for this specific setup, based on whether the
trend is intact or weakening.

### Target Reasoning
For each of T1/T2/T3, one sentence on what catalyst or condition would get price there.
What price action is the AI watching for between T1 and T2? Between T2 and T3?

### Position Sizing Sanity Check
One paragraph: is this position size reasonable? Use the risk:reward ratio and the
% of typical daily move (ATR) the trade requires. If the trade needs price to move
>2x the daily ATR to hit T2, flag that. If it can hit T2 within 1 ATR, say it's a
high-probability setup.

### Watch List (3-4 bullets)
- Specific events to monitor: earnings dates if approaching, F&O expiry day, sector news
- Price levels that would force re-evaluation BEFORE the stop hits
- Volume conditions that would confirm/invalidate the move
- {("If hedged: note any condition where the hedge becomes unnecessary" if direction == "hedged" else "If short: explicitly note short-squeeze risk and how to spot it" if direction == "short" else "Time decay: if the trade goes nowhere for N trading days, what's the rule?")}

### Important Caveats
- 2-3 honest risks that the algorithmic plan doesn't account for
- Acknowledge what's unknowable (gap risk, news shocks, etc.)

Tone: direct, no fluff, no hype, no "this is going to be great". Indian F&O trader,
serious about discipline. Use specific ₹ values when relevant. Maximum 750 words total."""


@app.route("/api/trade-plan", methods=["POST"])
def get_trade_plan():
    """
    Generate a concrete trade plan with algorithmic levels + AI commentary.

    Body:
      {
        "symbol":      "RELIANCE",
        "direction":   "long" | "short" | "hedged",
        "input_mode":  "qty" | "capital",
        "qty":         100,           # required if input_mode == "qty"
        "capital":     250000.0,      # required if input_mode == "capital"
      }
    """
    if not anthropic_client:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    body = request.get_json(silent=True) or {}

    # --- Input validation ---
    symbol = (body.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    if not all(c.isalnum() or c in "&-" for c in symbol):
        return jsonify({"error": "invalid symbol"}), 400

    direction = (body.get("direction") or "long").strip().lower()
    if direction not in ("long", "short", "hedged"):
        return jsonify({"error": "direction must be 'long', 'short', or 'hedged'"}), 400

    input_mode = (body.get("input_mode") or "qty").strip().lower()
    if input_mode not in ("qty", "capital"):
        return jsonify({"error": "input_mode must be 'qty' or 'capital'"}), 400

    qty = 0
    capital = 0.0
    if input_mode == "qty":
        try:
            qty = int(body.get("qty", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "qty must be an integer"}), 400
        if qty <= 0:
            return jsonify({"error": "qty must be > 0"}), 400
    else:
        try:
            capital = float(body.get("capital", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "capital must be a number"}), 400
        if capital <= 0:
            return jsonify({"error": "capital must be > 0"}), 400

    logger.info("Trade plan for %s (%s, %s=%s)",
                symbol, direction, input_mode, qty or capital)

    # --- Fetch market data in parallel ---
    price_data = _fetch_price_data(symbol)
    if price_data is None:
        return jsonify({"error": f"Could not fetch price data for {symbol}"}), 404

    current_price = price_data["current_price"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ratings = executor.submit(_fetch_analyst_ratings, symbol)
        future_options = executor.submit(fetch_options_data, symbol, current_price)
        ratings      = future_ratings.result()
        options_data = future_options.result()

    # Hedged direction requires options data — fail gracefully if unavailable
    if direction == "hedged" and not options_data:
        return jsonify({
            "error": "no_options_data",
            "message": (f"{symbol} is not in F&O segment — cannot hedge with a "
                        f"protective put. Choose Long or Short direction instead."),
        }), 400

    # --- Compute algorithmic levels ---
    levels = _compute_trade_levels(direction, price_data, options_data, ratings)

    # If user provided capital instead of qty, derive qty from capital + entry midpoint
    entry_mid = levels["entry_zone"]["mid"]
    if input_mode == "capital":
        qty = int(capital // entry_mid)
        if qty <= 0:
            return jsonify({
                "error": "capital_too_small",
                "message": (f"Capital ₹{capital:,.0f} is less than 1 share of "
                            f"{symbol} at ₹{entry_mid:.2f}. Increase capital or use qty input."),
            }), 400

    # --- Compute hedge leg (if hedged) ---
    hedge = None
    lot_size = _get_lot_size(symbol)
    if direction == "hedged":
        hedge = _compute_hedge_leg(current_price, qty, options_data)
        if hedge is None:
            return jsonify({
                "error": "no_atm_put",
                "message": (f"No ATM put with valid premium found for {symbol}. "
                            f"Try a different stock or non-hedged direction."),
            }), 400

    # --- Compute position economics ---
    econ = _compute_position_economics(levels, qty, lot_size, hedge)

    # --- AI commentary ---
    prompt = _build_trade_plan_prompt(
        symbol, direction, levels, econ, hedge,
        price_data, options_data, ratings,
    )

    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2400,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
    except Exception as e:
        logger.exception("Anthropic API call failed: %s", e)
        return jsonify({"error": f"AI commentary failed: {str(e)}"}), 500

    return jsonify({
        "symbol":         symbol,
        "analyzed_at":    datetime.utcnow().isoformat() + "Z",
        "direction":      direction,
        "input_mode":     input_mode,
        "lot_size":       lot_size,
        "levels":         levels,
        "economics":      econ,
        "hedge":          hedge,
        "ai_commentary":  ai_text,
        "model":          ANTHROPIC_MODEL,
    }), 200


# ===========================================================================
# PEER COMPARISON — find better alternatives aligned with the trade plan
# ===========================================================================
#
# Endpoint: POST /api/peer-comparison
# Body: { symbol, direction ("long"|"short"|"hedged") }
#
# Returns full data for the original symbol + 3 peer stocks, plus an AI
# verdict ranking them all and flagging which is the best fit for the
# selected trade direction.
#
# Peer selection: hardcoded PEER_MAP first, fall back to Yahoo sector tag.
# ===========================================================================

from peer_map import get_peers as _get_hardcoded_peers


def _resolve_peers(symbol: str, max_peers: int = 3) -> tuple[list[str], str]:
    """
    Find up to `max_peers` peer stocks for the given symbol.

    Strategy:
      1. Hardcoded peer map (fast, accurate for major stocks)
      2. Fall back to Yahoo sector matching (any stock in same NSE_STOCKS
         universe with same sector tag)

    Returns: (list of peer symbols, source string for logging/UI)
    """
    # Strategy 1: hardcoded map
    mapped = _get_hardcoded_peers(symbol)
    if mapped:
        return mapped[:max_peers], "hardcoded_map"

    # Strategy 2: sector-based fallback via Yahoo
    try:
        info = yf.Ticker(f"{symbol}.NS").info or {}
        my_sector = info.get("sector") or ""
        my_industry = info.get("industry") or ""
        if not (my_sector or my_industry):
            return [], "no_sector_data"

        # Iterate the F&O universe and find same-sector stocks
        candidates = []
        for other_sym in NSE_LOT_SIZES.keys():
            if other_sym == symbol:
                continue
            try:
                other_info = yf.Ticker(f"{other_sym}.NS").info or {}
                other_sector = other_info.get("sector") or ""
                other_industry = other_info.get("industry") or ""
                # Prefer industry match (tighter); fall back to sector match
                if my_industry and other_industry == my_industry:
                    candidates.append((other_sym, 2))   # priority 2 = same industry
                elif my_sector and other_sector == my_sector:
                    candidates.append((other_sym, 1))   # priority 1 = same sector only
                if len(candidates) >= max_peers * 2:
                    break  # don't over-search
            except Exception:
                continue

        candidates.sort(key=lambda c: -c[1])  # higher priority first
        return [c[0] for c in candidates[:max_peers]], "sector_fallback"

    except Exception as e:
        logger.warning("Peer resolution failed for %s: %s", symbol, e)
        return [], "error"


def _fetch_full_stock_bundle(symbol: str) -> Optional[dict]:
    """
    Fetch the same data set as /api/analyze for one symbol — price + options
    + analyst ratings + news. Returns None if price fetch fails.
    """
    price_data = _fetch_price_data(symbol)
    if price_data is None:
        return None

    current_price = price_data["current_price"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_ratings = executor.submit(_fetch_analyst_ratings, symbol)
        future_options = executor.submit(fetch_options_data, symbol, current_price)
        future_news    = executor.submit(_fetch_news, symbol, 3)

        ratings      = future_ratings.result()
        options_data = future_options.result()
        news         = future_news.result()

    # Serialize options for the response (same shape as /api/analyze)
    options_payload = None
    if options_data is not None:
        options_payload = {
            "expiry_date":       options_data.expiry_date,
            "pcr_oi":            round(options_data.pcr_oi, 2),
            "max_pain":          round(options_data.max_pain, 2),
            "pcr_signal":        options_data.pcr_signal,
            "buildup_signal":    options_data.buildup_signal,
            "composite_signal":  options_data.composite_signal,
            "composite_strength": options_data.composite_strength,
            "highest_call_oi_strike": options_data.highest_call_oi_strike,
            "highest_put_oi_strike":  options_data.highest_put_oi_strike,
        }

    return {
        "symbol":      symbol,
        "price":       price_data,
        "options":     options_payload,
        "ratings":     ratings,
        "news":        news,
    }


def _score_stock_for_direction(bundle: dict, direction: str) -> float:
    """
    Compute a simple alignment score (0-10) for how well this stock fits the
    chosen trade direction. AI gets this as a hint but is free to disagree.

    LONG-favoring signals (positive contribution):
      - RSI 40-65 (not overbought, room to run)
      - Price above SMA 50
      - Positive 30D change (momentum)
      - PCR > 1 (bullish options)
      - Composite signal bullish
      - Volume above average
      - Analyst consensus buy/strong buy
      - High Put OI support nearby

    SHORT-favoring signals: mirror image.
    HEDGED: same as LONG but penalize stocks with NO F&O (can't hedge).
    """
    p = bundle.get("price") or {}
    opt = bundle.get("options")
    ratings = bundle.get("ratings") or {}

    score = 5.0  # neutral baseline
    is_long_aligned = direction in ("long", "hedged")
    mult = 1 if is_long_aligned else -1

    # RSI alignment
    rsi = p.get("rsi_14") or 50
    if is_long_aligned:
        if 40 <= rsi <= 65: score += 1.0
        elif rsi > 70:      score -= 0.5
        elif rsi < 35:      score += 0.5   # oversold can mean bounce setup
    else:
        if 35 <= rsi <= 60: score += 1.0
        elif rsi < 30:      score -= 0.5
        elif rsi > 65:      score += 0.5   # overbought = short setup

    # Trend alignment (price vs SMA 50)
    price = p.get("current_price", 0)
    sma50 = p.get("sma_50", 0)
    if sma50 > 0 and price > 0:
        pct_from_sma = (price - sma50) / sma50 * 100
        score += mult * min(max(pct_from_sma / 5, -1.5), 1.5)  # clamp at ±1.5

    # Momentum (30D change)
    chg_30d = p.get("pct_change_30d") or 0
    score += mult * min(max(chg_30d / 15, -1.0), 1.0)  # clamp at ±1

    # Options sentiment
    if opt:
        comp = (opt.get("composite_signal") or "").lower()
        if "strong bull" in comp:    score += mult * 1.2
        elif "bull" in comp:         score += mult * 0.6
        elif "strong bear" in comp:  score -= mult * 1.2
        elif "bear" in comp:         score -= mult * 0.6

        pcr = opt.get("pcr_oi") or 1.0
        if is_long_aligned:
            if pcr > 1.3:   score += 0.6
            elif pcr < 0.7: score -= 0.6
        else:
            if pcr < 0.7:   score += 0.6
            elif pcr > 1.3: score -= 0.6
    elif direction == "hedged":
        # Can't hedge without options — heavy penalty
        score -= 2.0

    # Volume confirmation
    vol_ratio = p.get("volume_ratio") or 1
    if vol_ratio > 1.5:
        score += mult * 0.5

    # Analyst consensus
    consensus = (ratings.get("consensus") or "").lower()
    if "strong buy" in consensus:    score += mult * 0.8
    elif "buy" in consensus:         score += mult * 0.4
    elif "strong sell" in consensus: score -= mult * 0.8
    elif "sell" in consensus:        score -= mult * 0.4

    return round(max(0, min(10, score)), 2)


def _build_peer_comparison_prompt(
    target_bundle: dict, peer_bundles: list, direction: str, scores: dict,
) -> str:
    """Build AI prompt that ranks all stocks against the chosen direction."""
    direction_label = {
        "long":   "LONG (buy)",
        "short":  "SHORT (sell)",
        "hedged": "HEDGED LONG (buy stock + protective put)",
    }[direction]

    def _format_stock(bundle):
        p = bundle["price"]
        opt = bundle.get("options")
        ratings = bundle.get("ratings") or {}
        sym = bundle["symbol"]
        sc = scores.get(sym, "?")

        lines = [
            f"### {sym} (alignment score {sc}/10)",
            f"- Current price: ₹{p['current_price']:.2f}",
            f"- RSI(14): {p['rsi_14']:.1f}",
            f"- 1D / 30D: {p['pct_change_1d']:+.2f}% / {p['pct_change_30d']:+.2f}%",
            f"- 52w range: ₹{p['low_52w']:.0f} – ₹{p['high_52w']:.0f}",
            f"- SMA 50: ₹{p.get('sma_50', 0):.2f} (price is "
            f"{((p['current_price'] - p.get('sma_50', p['current_price'])) / max(p.get('sma_50', 1), 1) * 100):+.2f}% from SMA50)",
            f"- ATR(14): ₹{p.get('atr_14', 0):.2f}",
            f"- Volume: {p['volume_ratio']:.1f}x avg",
        ]
        if opt:
            lines += [
                f"- Options PCR: {opt['pcr_oi']:.2f} ({opt['pcr_signal']})",
                f"- Composite signal: {opt['composite_signal']} "
                f"({opt['composite_strength']}/5)",
                f"- Resistance / Support: ₹{opt['highest_call_oi_strike']:.0f} "
                f"/ ₹{opt['highest_put_oi_strike']:.0f}",
            ]
        else:
            lines.append("- Options: NOT IN F&O SEGMENT")

        if ratings and ratings.get("price_target"):
            pt = ratings["price_target"]
            tgt = pt.get("mean")
            cons = ratings.get("consensus", "—")
            if tgt:
                upside = (tgt - p['current_price']) / p['current_price'] * 100
                lines.append(f"- Analyst: {cons}, target ₹{tgt:.0f} ({upside:+.1f}%)")
            else:
                lines.append(f"- Analyst: {cons}")

        if bundle.get("news"):
            news_titles = " · ".join(n.get("title", "")[:80] for n in bundle["news"][:2])
            lines.append(f"- Recent news: {news_titles}")

        return "\n".join(lines)

    target_section = _format_stock(target_bundle)
    peer_sections = "\n\n".join(_format_stock(b) for b in peer_bundles)

    return f"""You are a senior portfolio manager helping a retail F&O trader compare
{target_bundle['symbol']} against its closest peers for a {direction_label} trade.

The trader is currently planning to take {direction_label} on {target_bundle['symbol']}.
Your job: rank ALL 4 stocks (target + peers) for how well each fits THIS specific
direction RIGHT NOW, given the data below.

## TARGET STOCK (currently chosen)

{target_section}

## PEER STOCKS (for comparison)

{peer_sections}

## YOUR TASK

Write the comparison memo in markdown with these EXACT sections:

### Verdict
ONE line at the top: which stock is the best {direction_label} setup RIGHT NOW?
Use this exact format:
**Best fit: SYMBOL** — one-line reason citing the strongest signal.

If the target stock {target_bundle['symbol']} is the best, say so clearly:
**Best fit: {target_bundle['symbol']} (your selection)** — and explain why.

If a peer is better, say:
**Best fit: SYMBOL (better than your selection of {target_bundle['symbol']})** — and
explain WHY the peer is better.

### Ranked Comparison
Rank all 4 stocks 1-4 for {direction_label} alignment. For each:
- Show the alignment score
- 2 sentences explaining what's working / what's not
- One specific data point that drives the rank (e.g., "RSI 78 is too hot for entry here")

### Why {target_bundle['symbol']} {"wins" if target_bundle else "loses"} vs peers
Honest 3-4 sentence assessment:
- If {target_bundle['symbol']} ranks #1: confirm what makes it the best pick
- If it doesn't: state plainly which peer is stronger and on what metric

### Setup Differences That Matter
Where the peers diverge from {target_bundle['symbol']} in ways that affect THIS trade:
- One bullet per material difference (RSI, trend stage, options sentiment, valuation)
- Max 4 bullets

### Switching Considerations
Should the trader actually switch from {target_bundle['symbol']} to the top-ranked peer?
- If yes: what specifically would you do differently in the trade plan
- If no: why staying with {target_bundle['symbol']} is still defensible

### Honest Caveats
- 2 bullets: what this comparison DOESN'T capture (e.g., position already opened,
  tax implications, sector concentration in user's portfolio)
- Acknowledge analyst data may be sparse for some peers

Tone: direct, decisive, no fluff. Use specific numbers from the data. If the target
stock IS the best pick, don't manufacture reasons to switch. If a peer is genuinely
better, say so clearly. Max 700 words."""


@app.route("/api/peer-comparison", methods=["POST"])
def get_peer_comparison():
    """
    Compare the target stock against 3 peer stocks for a chosen direction.

    Body:
      {
        "symbol":    "HDFCBANK",
        "direction": "long" | "short" | "hedged",
      }
    """
    if not anthropic_client:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    body = request.get_json(silent=True) or {}

    symbol = (body.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    if not all(c.isalnum() or c in "&-" for c in symbol):
        return jsonify({"error": "invalid symbol"}), 400

    direction = (body.get("direction") or "long").strip().lower()
    if direction not in ("long", "short", "hedged"):
        return jsonify({"error": "direction must be 'long', 'short', or 'hedged'"}), 400

    logger.info("Peer comparison for %s (%s)", symbol, direction)

    # --- Resolve peers ---
    peer_symbols, source = _resolve_peers(symbol, max_peers=3)
    logger.info("Peers for %s: %s (source: %s)", symbol, peer_symbols, source)

    if not peer_symbols:
        return jsonify({
            "error":   "no_peers_found",
            "message": (f"Could not find peer stocks for {symbol}. "
                        f"Not in our peer map and Yahoo sector lookup also failed."),
        }), 404

    # --- Fetch target + peers in parallel ---
    # We use ThreadPoolExecutor with 4 workers (target + up to 3 peers).
    # Each _fetch_full_stock_bundle is itself parallel (3 workers internally for
    # options/ratings/news), so total worker count is 4 * 3 = 12 — manageable.
    all_symbols = [symbol] + peer_symbols
    bundles: dict = {}

    with ThreadPoolExecutor(max_workers=len(all_symbols)) as executor:
        futures = {executor.submit(_fetch_full_stock_bundle, s): s for s in all_symbols}
        for future in futures:
            sym = futures[future]
            try:
                result = future.result()
                if result is not None:
                    bundles[sym] = result
            except Exception as e:
                logger.warning("Bundle fetch failed for %s: %s", sym, e)

    if symbol not in bundles:
        return jsonify({
            "error":   "target_fetch_failed",
            "message": f"Could not fetch data for {symbol}",
        }), 404

    target_bundle = bundles[symbol]
    peer_bundles  = [bundles[s] for s in peer_symbols if s in bundles]

    if not peer_bundles:
        return jsonify({
            "error":   "all_peers_failed",
            "message": (f"Could not fetch data for any of the peer stocks: {peer_symbols}. "
                        f"Yahoo may be rate-limiting."),
        }), 503

    # --- Score each stock for the chosen direction ---
    scores = {s: _score_stock_for_direction(bundles[s], direction) for s in bundles}

    # --- AI verdict ---
    prompt = _build_peer_comparison_prompt(target_bundle, peer_bundles, direction, scores)

    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2400,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
    except Exception as e:
        logger.exception("Anthropic API call failed: %s", e)
        return jsonify({"error": f"AI verdict failed: {str(e)}"}), 500

    # --- Serialize the response ---
    def _summarize_bundle(b):
        p = b["price"]
        opt = b.get("options")
        ratings = b.get("ratings") or {}
        return {
            "symbol":          b["symbol"],
            "alignment_score": scores.get(b["symbol"], 0),
            "current_price":   p["current_price"],
            "pct_change_1d":   p["pct_change_1d"],
            "pct_change_30d":  p["pct_change_30d"],
            "rsi_14":          p["rsi_14"],
            "volume_ratio":    p["volume_ratio"],
            "high_52w":        p["high_52w"],
            "low_52w":         p["low_52w"],
            "sma_50":          p.get("sma_50"),
            "atr_14":          p.get("atr_14"),
            "options":         opt,
            "consensus":       ratings.get("consensus"),
            "price_target":    ratings.get("price_target"),
        }

    return jsonify({
        "target":         symbol,
        "direction":      direction,
        "peer_source":    source,
        "analyzed_at":    datetime.utcnow().isoformat() + "Z",
        "target_bundle":  _summarize_bundle(target_bundle),
        "peer_bundles":   [_summarize_bundle(b) for b in peer_bundles],
        "scores":         scores,
        "ai_verdict":     ai_text,
        "model":          ANTHROPIC_MODEL,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
