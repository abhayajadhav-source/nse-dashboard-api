"""
NSE Dashboard API — read-only JSON for the Cloudflare Pages dashboard,
plus on-demand stock analysis via Anthropic Claude.

Endpoints:
  GET  /                  → health check
  GET  /api/snapshot      → latest scanner snapshots
  POST /api/analyze       → analyze a single stock with Claude
                            body: {"symbol": "RELIANCE"}
"""

from __future__ import annotations

import logging
import os
import urllib.parse
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

from technical_indicators import atr, bollinger_bands, ema, macd, rsi, sma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

DATABASE_URL      = os.getenv("DATABASE_URL", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

ANTHROPIC_MODEL = "claude-haiku-4-5"

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
# Existing endpoints
# ---------------------------------------------------------------------------
@app.route("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "nse-dashboard-api",
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
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
# Analyst ratings (NEW)
# ---------------------------------------------------------------------------
def _fetch_analyst_ratings(symbol: str) -> dict:
    """
    Fetch analyst recommendations and price targets from Yahoo Finance.

    Returns a dict with:
      - summary:     aggregated counts {strong_buy, buy, hold, sell, strong_sell, total}
      - consensus:   short label ('Buy', 'Hold', 'Sell', etc.) based on majority
      - price_target: {mean, high, low, count, currency}
      - recent_actions: list of recent broker actions (upgrades/downgrades/initiations)
      - data_source:  'yfinance' or 'unavailable'

    Returns {} on failure so the caller can degrade gracefully.
    """
    yf_symbol = f"{symbol}.NS"
    result = {
        "summary":         None,
        "consensus":       None,
        "price_target":    None,
        "recent_actions":  [],
        "data_source":     "unavailable",
    }

    try:
        ticker = yf.Ticker(yf_symbol)
        info   = ticker.info or {}

        # --- Summary counts (Yahoo aggregates these) ---
        # Yahoo exposes a recommendationKey and numberOfAnalystOpinions
        rec_key   = info.get("recommendationKey")   # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell', 'none'
        rec_mean  = info.get("recommendationMean")  # 1.0 (strong buy) -> 5.0 (strong sell)
        num_anlst = info.get("numberOfAnalystOpinions")

        # Try the detailed breakdown if available
        try:
            rec_df = ticker.recommendations_summary
            if rec_df is not None and not rec_df.empty:
                # First row is the most recent period (current month aggregate)
                row = rec_df.iloc[0]
                result["summary"] = {
                    "strong_buy":  int(row.get("strongBuy",  0) or 0),
                    "buy":         int(row.get("buy",        0) or 0),
                    "hold":        int(row.get("hold",       0) or 0),
                    "sell":        int(row.get("sell",       0) or 0),
                    "strong_sell": int(row.get("strongSell", 0) or 0),
                }
                result["summary"]["total"] = sum(result["summary"].values())
        except Exception as e:
            logger.debug("recommendations_summary unavailable for %s: %s", symbol, e)

        # Fall back to top-level fields if breakdown failed
        if result["summary"] is None and num_anlst:
            # We have a count but no breakdown — best we can do is store the count
            result["summary"] = {
                "strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0,
                "total": int(num_anlst),
            }

        # Map Yahoo's recommendationKey to a clean label
        if rec_key and rec_key != "none":
            label_map = {
                "strong_buy":  "Strong Buy",
                "buy":         "Buy",
                "hold":        "Hold",
                "sell":        "Sell",
                "strong_sell": "Strong Sell",
                "underperform":"Underperform",
                "outperform":  "Outperform",
            }
            result["consensus"] = label_map.get(rec_key, rec_key.replace("_", " ").title())
        elif rec_mean:
            # Derive label from mean if key missing (1=Strong Buy, 5=Strong Sell)
            if   rec_mean <= 1.5: result["consensus"] = "Strong Buy"
            elif rec_mean <= 2.5: result["consensus"] = "Buy"
            elif rec_mean <= 3.5: result["consensus"] = "Hold"
            elif rec_mean <= 4.5: result["consensus"] = "Sell"
            else:                 result["consensus"] = "Strong Sell"

        # --- Price targets (analyst consensus targets) ---
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

        # --- Recent broker actions (upgrades/downgrades/initiations) ---
        try:
            upgrades_df = ticker.upgrades_downgrades
            if upgrades_df is not None and not upgrades_df.empty:
                # Take the 8 most recent actions
                # Index is GradeDate; columns include Firm, ToGrade, FromGrade, Action
                recent = upgrades_df.head(8)
                for idx, row in recent.iterrows():
                    # idx is the action date
                    try:
                        action_date = pd.to_datetime(idx).strftime("%d %b %Y")
                    except Exception:
                        action_date = str(idx)[:10]
                    result["recent_actions"].append({
                        "date":       action_date,
                        "firm":       str(row.get("Firm", "") or ""),
                        "to_grade":   str(row.get("ToGrade", "") or ""),
                        "from_grade": str(row.get("FromGrade", "") or ""),
                        "action":     str(row.get("Action", "") or ""),  # 'up', 'down', 'init', 'main'
                    })
        except Exception as e:
            logger.debug("upgrades_downgrades unavailable for %s: %s", symbol, e)

        # Mark as success if we got anything useful
        if (result["summary"] or result["consensus"] or
            result["price_target"] or result["recent_actions"]):
            result["data_source"] = "yfinance"

    except Exception as e:
        logger.warning("Analyst rating fetch failed for %s: %s", symbol, e)

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
# Prompt builder — now includes analyst ratings
# ---------------------------------------------------------------------------
def _format_ratings_for_prompt(ratings: dict, current_price: float) -> str:
    """Compact text block summarising analyst data, fed into the Claude prompt."""
    if ratings.get("data_source") == "unavailable":
        return "  (no analyst data available from Yahoo Finance)"

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
        else:
            lines.append(f"  {s['total']} analysts covering (no breakdown available)")

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
            action_label = (a.get("action") or "").lower()
            arrow = ""
            if   action_label in ("up", "upgrade"):     arrow = "↑"
            elif action_label in ("down", "downgrade"): arrow = "↓"
            elif action_label in ("init",):             arrow = "★"
            elif action_label in ("main",):             arrow = "·"
            lines.append(f"    {arrow} {a['date']} — {a['firm']}{transition}")

    return "\n".join(lines) if lines else "  (no analyst data available)"


def _build_analysis_prompt(price_data: dict, news: list, ratings: dict) -> str:
    p = price_data

    news_block = ""
    if news:
        news_lines = []
        for n in news:
            age = "today" if n["age_days"] == 0 else f"{n['age_days']}d ago"
            news_lines.append(f"  - [{age}] {n['title']} ({n['source']})")
        news_block = "\n".join(news_lines)
    else:
        news_block = "  (no recent headlines found in last 7 days)"

    ratings_block = _format_ratings_for_prompt(ratings, p["current_price"])

    return f"""You are a technical analyst for Indian (NSE) stocks. Analyze the following daily-timeframe data for {p['symbol']} and provide a balanced, actionable assessment.

PRICE & VOLUME
- Current price: ₹{p['current_price']:,.2f}
- Today's change: {p['pct_change_1d']:+.2f}%
- 5-day change: {p['pct_change_5d']:+.2f}%
- 30-day change: {p['pct_change_30d']:+.2f}%
- 90-day change: {p['pct_change_90d']:+.2f}%
- Today's range: ₹{p['day_low']:,.2f} – ₹{p['day_high']:,.2f}
- Volume today: {p['volume']:,} (vs 20d avg {p['volume_20d_avg']:,} → {p['volume_ratio']}x)

52-WEEK RANGE
- 52w high: ₹{p['high_52w']:,.2f} (currently {p['pct_from_52w_high']:+.2f}% from high)
- 52w low:  ₹{p['low_52w']:,.2f}  (currently {p['pct_from_52w_low']:+.2f}% from low)

TREND (Moving Averages)
- SMA 20:  ₹{p['sma_20']:,.2f}  ({'above' if p['current_price'] > p['sma_20']  else 'below'})
- SMA 50:  ₹{p['sma_50']:,.2f}  ({'above' if p['current_price'] > p['sma_50']  else 'below'})
- SMA 200: ₹{p['sma_200']:,.2f} ({'above' if p['current_price'] > p['sma_200'] else 'below'})
- EMA 9:   ₹{p['ema_9']:,.2f}
- EMA 21:  ₹{p['ema_21']:,.2f}

MOMENTUM
- RSI (14): {p['rsi_14']} {'(overbought)' if p['rsi_14'] > 70 else '(oversold)' if p['rsi_14'] < 30 else '(neutral)'}
- MACD line: {p['macd_line']}
- MACD signal: {p['macd_signal']}
- MACD histogram: {p['macd_hist']} {'(bullish)' if p['macd_hist'] > 0 else '(bearish)'}

VOLATILITY / BANDS
- Bollinger upper: ₹{p['bb_upper']:,.2f}
- Bollinger middle: ₹{p['bb_middle']:,.2f}
- Bollinger lower: ₹{p['bb_lower']:,.2f}
- ATR (14): ₹{p['atr_14']:,.2f}

ANALYST RATINGS & PRICE TARGETS
{ratings_block}

RECENT NEWS HEADLINES (last 7 days)
{news_block}

INSTRUCTIONS
Provide your analysis in exactly this structure, using Markdown:

## Trend Summary
2-3 sentences describing current trend direction (bullish/bearish/sideways), strength, and conviction level.

## Key Technical Observations
- Bullet points (4-6) covering the most important indicator signals
- Be specific with numbers; tie observations to actual values above
- Cover MA structure, RSI/MACD, volatility, and volume

## Support & Resistance
- Immediate support: ₹X (rationale)
- Immediate resistance: ₹Y (rationale)
- Use recent pivots, MAs, BB bands, and 52w levels

## Analyst View vs Technical View
- 1-2 sentences comparing what brokers are saying (rating + price target) with what the technicals show
- Flag any disagreement (e.g., "brokers bullish but RSI overbought" or "technical breakdown despite buy ratings")
- If no analyst data is available, say so and skip this section

## News Context
1-2 sentences on how the recent headlines (if any) align with or contradict the technicals.

## Recommendation
One of: **STRONG BUY**, **BUY**, **HOLD**, **SELL**, **STRONG SELL**, **AVOID**

Follow with:
- Time horizon (intraday / swing 1-4 weeks / positional 1-3 months)
- Entry zone (if buying) or stop-loss / exit zone (if selling)
- Target (realistic, based on resistance levels and analyst targets where relevant)
- Risk-reward ratio

## Risks & Caveats
2-3 bullet points covering what could invalidate this view.

IMPORTANT GUARDRAILS
- Be honest about uncertainty. If signals conflict, say so.
- Do not predict prices with false precision.
- Use analyst targets as one input among many — they're often lagging.
- End with: "*This is technical analysis only. Not investment advice. Past performance does not guarantee future returns.*"
"""


# ---------------------------------------------------------------------------
# Analyze endpoint
# ---------------------------------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze_stock():
    if not anthropic_client:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    data   = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required in body"}), 400

    if not all(c.isalnum() or c in "&-" for c in symbol):
        return jsonify({"error": "invalid symbol format"}), 400

    logger.info("Analyzing %s", symbol)

    # 1. Fetch price data + indicators
    price_data = _fetch_price_data(symbol)
    if price_data is None:
        return jsonify({
            "error": f"Could not fetch data for {symbol}. "
                     "Check the symbol exists on NSE."
        }), 404

    # 2. Fetch analyst ratings (NEW)
    ratings = _fetch_analyst_ratings(symbol)

    # 3. Fetch recent news
    news = _fetch_news(symbol)

    # 4. Build prompt and call Claude
    prompt = _build_analysis_prompt(price_data, news, ratings)
    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1800,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
    except Exception as e:
        logger.exception("Anthropic API call failed: %s", e)
        return jsonify({"error": f"AI analysis failed: {str(e)}"}), 500

    return jsonify({
        "symbol":      symbol,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "price_data":  price_data,
        "ratings":     ratings,
        "news":        news,
        "analysis":    analysis_text,
        "model":       ANTHROPIC_MODEL,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
