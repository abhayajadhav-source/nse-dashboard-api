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
      2. Otherwise, call yfinance
      3. On success, save to cache and return fresh data
      4. On rate-limit / failure, return empty result (cold-start behavior:
         show 'no data' rather than ancient cached entries)
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

        rec_key   = info.get("recommendationKey")
        rec_mean  = info.get("recommendationMean")
        num_anlst = info.get("numberOfAnalystOpinions")

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
        except Exception:
            pass

        if result["summary"] is None and num_anlst:
            result["summary"] = {
                "strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0,
                "total": int(num_anlst),
            }

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
        except Exception:
            pass

        if (result["summary"] or result["consensus"] or
            result["price_target"] or result["recent_actions"]):
            result["data_source"] = "yfinance"

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

    # Convert option chain to StrategyContext format
    strikes = []
    for s in (options_data.top_strikes or []):
        strikes.append(OptionStrike(
            strike=s["strike"],
            call_ltp=s.get("call_ltp", 0.0),
            call_oi=s.get("call_oi", 0),
            put_ltp=s.get("put_ltp", 0.0),
            put_oi=s.get("put_oi", 0),
            distance_from_spot=s.get("distance_from_spot", 0.0),
        ))
    strikes.sort(key=lambda s: s.strike)

    if len(strikes) < 5:
        return jsonify({
            "error": "insufficient_strikes",
            "message": f"Only {len(strikes)} strikes available — need 5+ for strategy analysis."
        }), 404

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
