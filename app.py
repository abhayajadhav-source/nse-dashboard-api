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

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
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

# ---------------------------------------------------------------------------
# CORS — lock to your own frontend origin(s) instead of "*".
# Set ALLOWED_ORIGINS env var to a comma-separated list, e.g.
#   https://nse-dashboard.abhay-a-jadhav.workers.dev
# Falls back to "*" only if the var is unset (so nothing breaks before you
# configure it — but you SHOULD set it once deployed).
# ---------------------------------------------------------------------------
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if _origins_env:
    _allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    _allowed_origins = ["*"]
CORS(app, origins=_allowed_origins)

DATABASE_URL      = os.getenv("DATABASE_URL", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Password login + signed-token auth (Path 3)
# ---------------------------------------------------------------------------
# How it works:
#   1. User POSTs the password to /api/login
#   2. If correct, we issue an HMAC-signed token (like a minimal JWT) that
#      encodes an expiry timestamp. The token is signed with AUTH_SIGNING_SECRET
#      so it can't be forged.
#   3. The frontend stores the token and sends it as "Authorization: Bearer <token>"
#      on every subsequent request.
#   4. before_request verifies the signature + expiry on protected endpoints.
#
# The password itself is NEVER stored in the frontend — only the derived,
# expiring token. Viewing page source reveals nothing useful.
#
# Two env vars on Render:
#   AUTH_PASSWORD        — the login password (plain text; compared in constant time)
#   AUTH_SIGNING_SECRET  — a long random string used to sign tokens
#
# If AUTH_PASSWORD is unset, the gate FAILS OPEN (disabled) so you can deploy
# the code first and configure the password second without locking yourself out.
# ---------------------------------------------------------------------------
AUTH_PASSWORD       = os.getenv("AUTH_PASSWORD", "").strip()
AUTH_SIGNING_SECRET = os.getenv("AUTH_SIGNING_SECRET", "").strip()
# If no signing secret is configured, generate an ephemeral one at boot. This
# means tokens survive only until the next restart when the secret is unset —
# fine for fail-open mode, but you SHOULD set AUTH_SIGNING_SECRET in production
# so tokens persist across Render restarts/sleeps.
if not AUTH_SIGNING_SECRET:
    AUTH_SIGNING_SECRET = secrets.token_urlsafe(48)

TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60   # 30-day session

# Paths that never require a token (health check, login itself, preflight)
_AUTH_EXEMPT_PATHS = {"/", "/health", "/api/login"}


def _b64url_encode(raw: bytes) -> str:
    """URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64url_decode(s: str) -> bytes:
    """Reverse of _b64url_encode — re-add padding before decoding."""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _issue_token(ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    """
    Create an HMAC-signed token: base64(payload).base64(signature)
    payload = {"exp": <unix expiry>, "iat": <unix issued-at>}
    """
    payload = {"exp": int(time.time()) + ttl_seconds, "iat": int(time.time())}
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(
        AUTH_SIGNING_SECRET.encode(), payload_b64.encode(), hashlib.sha256
    ).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{payload_b64}.{sig_b64}"


def _verify_token(token: str) -> bool:
    """Verify token signature + expiry. Returns True if valid."""
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        # Recompute the signature and compare in constant time
        expected_sig = hmac.new(
            AUTH_SIGNING_SECRET.encode(), payload_b64.encode(), hashlib.sha256
        ).digest()
        provided_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, provided_sig):
            return False
        # Signature valid — now check expiry
        payload = json.loads(_b64url_decode(payload_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            return False
        return True
    except Exception:
        return False


@app.before_request
def _require_auth():
    """Reject any request lacking a valid Bearer token on protected paths."""
    # Always allow CORS preflight
    if request.method == "OPTIONS":
        return None
    # Allow exempt paths (health check, login)
    if request.path in _AUTH_EXEMPT_PATHS:
        return None
    # Fail OPEN if no password configured (lets you deploy before configuring)
    if not AUTH_PASSWORD:
        return None
    # Otherwise require a valid Bearer token
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "unauthorized — login required"}), 401
    token = auth_header[len("Bearer "):].strip()
    if not _verify_token(token):
        return jsonify({"error": "unauthorized — invalid or expired session"}), 401
    return None


# Simple in-memory brute-force guard for the login endpoint.
# Tracks failed attempts per IP; locks out after too many within a window.
# Resets on restart (fine for single-user). Not decorator-based because the
# limiter object is defined later in the file.
_login_attempts: dict[str, list[float]] = {}
_LOGIN_MAX_ATTEMPTS = 8           # max failed attempts...
_LOGIN_WINDOW_SECONDS = 15 * 60   # ...within this rolling window
_LOGIN_LOCKOUT_SECONDS = 15 * 60  # lockout duration once exceeded


def _login_rate_ok(ip: str) -> bool:
    """Return True if this IP is allowed another login attempt."""
    now = time.time()
    attempts = _login_attempts.get(ip, [])
    # Drop attempts outside the window
    attempts = [t for t in attempts if now - t < _LOGIN_WINDOW_SECONDS]
    _login_attempts[ip] = attempts
    return len(attempts) < _LOGIN_MAX_ATTEMPTS


def _record_login_failure(ip: str) -> None:
    _login_attempts.setdefault(ip, []).append(time.time())


@app.route("/api/login", methods=["POST"])
def login():
    """
    Exchange the password for a signed session token.

    Body: { "password": "..." }
    Returns: { "token": "...", "expires_in": <seconds> }
    """
    # If auth is disabled (no password set), issue a token anyway so the
    # frontend flow works uniformly.
    if not AUTH_PASSWORD:
        return jsonify({
            "token": _issue_token(),
            "expires_in": TOKEN_TTL_SECONDS,
            "auth_disabled": True,
        }), 200

    # Brute-force guard
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if not _login_rate_ok(ip):
        return jsonify({
            "error": "too many failed attempts — wait 15 minutes and try again"
        }), 429

    body = request.get_json(silent=True) or {}
    provided = (body.get("password") or "")
    # Constant-time comparison to avoid timing attacks
    if not hmac.compare_digest(provided, AUTH_PASSWORD):
        _record_login_failure(ip)
        return jsonify({"error": "incorrect password"}), 401

    # Success — clear this IP's failure history
    _login_attempts.pop(ip, None)
    return jsonify({
        "token": _issue_token(),
        "expires_in": TOKEN_TTL_SECONDS,
    }), 200


# ---------------------------------------------------------------------------
# Rate limiting (Flask-Limiter) — caps damage from runaway scripts even if
# the API key leaks. Limits are per-IP.
#   - AI / expensive endpoints: 50/hour
#   - Read-only endpoints:      200/hour
# Uses in-memory storage (resets on Render restart — fine for a single dyno).
# ---------------------------------------------------------------------------
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per hour"],   # default for any endpoint not decorated
        storage_uri="memory://",
    )
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    # flask_limiter not installed yet — define a no-op decorator so the app
    # still boots. Add 'Flask-Limiter' to requirements.txt to enable.
    logger.warning(
        "flask_limiter not installed — rate limiting disabled. "
        "Add 'Flask-Limiter' to requirements.txt to enable."
    )
    RATE_LIMIT_AVAILABLE = False

    class _NoOpLimiter:
        def limit(self, *a, **k):
            def deco(f):
                return f
            return deco
    limiter = _NoOpLimiter()

# Convenience decorators — apply to routes below
AI_RATE_LIMIT   = "50 per hour"    # expensive: AI-backed endpoints
READ_RATE_LIMIT = "200 per hour"   # cheaper: read-only data

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
# Market context endpoint — powers the sticky symbol bar
# ---------------------------------------------------------------------------
# Returns NIFTY 50, BANK NIFTY, and India VIX prices + 1-day % change.
# Lightweight (just 3 yfinance lookups) and cached for 60s via Yahoo.
# Used by the persistent header bar so every screen shows market context.
# ---------------------------------------------------------------------------
_MARKET_CONTEXT_CACHE = {"data": None, "expires_at": 0}
_MARKET_CONTEXT_TTL_SECONDS = 60   # 60-second in-memory cache


@app.route("/api/market-context")
def get_market_context():
    """Return current NIFTY 50, BANK NIFTY, and India VIX levels."""
    import time
    now = time.time()

    # Serve from in-memory cache if fresh (avoids hammering Yahoo)
    if _MARKET_CONTEXT_CACHE["data"] and now < _MARKET_CONTEXT_CACHE["expires_at"]:
        return jsonify({**_MARKET_CONTEXT_CACHE["data"], "cached": True}), 200

    # Yahoo Finance tickers for Indian indices
    tickers = {
        "nifty":      "^NSEI",
        "bank_nifty": "^NSEBANK",
        "india_vix":  "^INDIAVIX",
    }

    out = {}
    for key, yf_ticker in tickers.items():
        try:
            t = yf.Ticker(yf_ticker)
            # Use 2-day history to compute today's change from yesterday's close
            hist = t.history(period="2d", interval="1d")
            if hist.empty or len(hist) < 1:
                out[key] = None
                continue
            current = float(hist["Close"].iloc[-1])
            # If we have at least 2 days, compute the change vs prior close
            if len(hist) >= 2:
                prior = float(hist["Close"].iloc[-2])
                change_abs = current - prior
                change_pct = (change_abs / prior) * 100.0 if prior else 0.0
            else:
                change_abs = 0.0
                change_pct = 0.0
            out[key] = {
                "value":      round(current, 2),
                "change_abs": round(change_abs, 2),
                "change_pct": round(change_pct, 2),
            }
        except Exception as e:
            logger.warning("Failed to fetch market context for %s: %s", yf_ticker, e)
            out[key] = None

    # Tag the VIX regime so the frontend can show a clear label
    vix_value = out.get("india_vix", {}).get("value") if out.get("india_vix") else None
    if vix_value is None:
        vix_regime = "unknown"
    elif vix_value < 12:
        vix_regime = "low"          # Complacent market, mean-reverting strategies risky
    elif vix_value < 18:
        vix_regime = "normal"
    elif vix_value < 25:
        vix_regime = "elevated"
    else:
        vix_regime = "spiked"       # Volatile market, expect wider swings
    out["vix_regime"] = vix_regime

    out["fetched_at"] = datetime.utcnow().isoformat() + "Z"
    out["cached"] = False

    # Store in cache
    _MARKET_CONTEXT_CACHE["data"] = out
    _MARKET_CONTEXT_CACHE["expires_at"] = now + _MARKET_CONTEXT_TTL_SECONDS

    return jsonify(out), 200


# ---------------------------------------------------------------------------
# Risk Dashboard — portfolio-level view of open journal positions
# ---------------------------------------------------------------------------
# Reads all OPEN trades from trading_journal, fetches live spot + beta from
# Yahoo for each, then computes:
#   - Exposure: capital deployed (entry-based) vs current value (MTM)
#   - Concentration: per-symbol %, per-sector %, long/short skew
#   - Stress test: portfolio impact for NIFTY moves -10/-5/-2/0/+2/+5%
#
# Beta is fetched per-symbol from Yahoo (5-year against the index). Falls
# back to 1.0 if missing. Parallel fetches via ThreadPoolExecutor keep
# the endpoint usable even with many positions.
# ---------------------------------------------------------------------------

def _fetch_symbol_risk_data(symbol: str) -> dict:
    """
    Fetch live price + beta + sector for a single symbol.
    Best-effort: returns sensible defaults on any failure rather than raising.
    """
    yf_symbol = f"{symbol}.NS"
    out = {
        "current_price": None,
        "beta":          1.0,    # default to market-beta if unknown
        "sector":        "Unknown",
    }
    try:
        t = yf.Ticker(yf_symbol)
        info = t.info or {}
        # Current price — try multiple keys, Yahoo's shape varies
        price = (info.get("currentPrice")
                 or info.get("regularMarketPrice")
                 or info.get("previousClose"))
        if price:
            out["current_price"] = float(price)
        # Beta (5-year vs NIFTY for .NS tickers)
        beta = info.get("beta")
        if beta is not None and not (isinstance(beta, float) and beta != beta):  # NaN check
            out["beta"] = round(float(beta), 2)
        # Sector — useful for concentration breakdown
        sector = info.get("sector")
        if sector:
            out["sector"] = sector
    except Exception as e:
        logger.warning("Risk data fetch failed for %s: %s", symbol, e)
    return out


def _classify_concentration(pct: float, kind: str) -> Optional[str]:
    """
    Return a warning level for a given concentration percentage.
    kind: 'symbol' | 'sector' | 'direction' | 'deployment'
    """
    if kind == "symbol":
        if pct > 30: return "danger"   # >30% in one name is dangerous
        if pct > 20: return "warning"  # >20% is concerning
        return None
    if kind == "sector":
        if pct > 50: return "danger"
        if pct > 35: return "warning"
        return None
    if kind == "direction":
        # Direction skew: pct here is the ABS difference between long & short
        if pct > 80: return "warning"  # >80% one-sided
        return None
    if kind == "deployment":
        if pct > 100: return "danger"  # over-deployed (leveraged)
        if pct > 80:  return "warning"
        return None
    return None


@app.route("/api/risk-dashboard")
@limiter.limit(READ_RATE_LIMIT)
def get_risk_dashboard():
    """Compute portfolio-level risk view across all open journal positions."""
    # journal_store is imported later in the file; do a lazy import here so
    # this endpoint definition stays self-contained and the module loads
    # even if journal_store has a problem.
    try:
        import journal_store
    except Exception as e:
        logger.exception("Risk dashboard: journal_store unavailable")
        return jsonify({"error": f"Journal module unavailable: {e}"}), 500

    try:
        # Fetch all open positions from the journal
        positions_raw = journal_store.list_trades(status="open", limit=200)
    except Exception as e:
        logger.exception("Risk dashboard: failed to list open trades")
        return jsonify({"error": f"Failed to load open positions: {e}"}), 500

    if not positions_raw:
        # Empty book — still return a valid response so frontend can render "no positions" state
        return jsonify({
            "as_of":         datetime.utcnow().isoformat() + "Z",
            "positions":     [],
            "summary": {
                "total_positions":      0,
                "long_positions":       0,
                "short_positions":      0,
                "hedged_positions":     0,
                "capital_at_entry":     0,
                "current_value":        0,
                "total_unrealized_pnl": 0,
                "total_unrealized_pct": 0,
                "long_exposure":        0,
                "short_exposure":       0,
                "net_exposure":         0,
            },
            "concentration": {
                "by_symbol":    [],
                "by_sector":    [],
                "by_direction": {"long_pct": 0, "short_pct": 0, "hedged_pct": 0, "skew": "empty"},
            },
            "stress_test":   {"scenarios": []},
            "warnings":      [],
        }), 200

    # Parallel-fetch live data for all symbols
    # Dedupe — if user has 2 RELIANCE positions, only fetch RELIANCE once
    unique_symbols = list(set(p["symbol"] for p in positions_raw))
    symbol_data = {}
    with ThreadPoolExecutor(max_workers=min(8, len(unique_symbols))) as executor:
        future_map = {executor.submit(_fetch_symbol_risk_data, s): s for s in unique_symbols}
        for future in future_map:
            sym = future_map[future]
            try:
                symbol_data[sym] = future.result(timeout=10)
            except Exception as e:
                logger.warning("Risk data future failed for %s: %s", sym, e)
                symbol_data[sym] = {"current_price": None, "beta": 1.0, "sector": "Unknown"}

    # Enrich each position with live data + per-position metrics
    enriched = []
    for p in positions_raw:
        sym = p["symbol"]
        live = symbol_data.get(sym, {"current_price": None, "beta": 1.0, "sector": "Unknown"})
        qty   = int(p.get("qty") or 0)
        entry = float(p.get("entry_price") or 0)
        cur   = live["current_price"] if live["current_price"] else entry
        direction = (p.get("direction") or "long").lower()

        capital_at_entry = qty * entry
        current_value    = qty * cur
        # Unrealized P&L respects direction sign
        if direction == "short":
            unrealized = (entry - cur) * qty
        else:  # long or hedged — both have positive delta exposure
            unrealized = (cur - entry) * qty
        unrealized_pct = (unrealized / capital_at_entry * 100) if capital_at_entry else 0

        enriched.append({
            "id":                 p.get("id"),
            "symbol":             sym,
            "direction":          direction,
            "qty":                qty,
            "entry_price":        entry,
            "current_price":      cur,
            "sector":             live["sector"],
            "beta":               live["beta"],
            "capital_at_entry":   round(capital_at_entry, 2),
            "current_value":      round(current_value, 2),
            "unrealized_pnl":     round(unrealized, 2),
            "unrealized_pnl_pct": round(unrealized_pct, 2),
            "entry_date":         p.get("entry_date"),
            "stop_loss":          p.get("stop_loss"),
        })

    # ----- Aggregate summary -----
    total_capital_entry  = sum(p["capital_at_entry"] for p in enriched)
    total_current_value  = sum(p["current_value"] for p in enriched)
    total_unrealized     = sum(p["unrealized_pnl"] for p in enriched)
    long_exposure  = sum(p["current_value"] for p in enriched if p["direction"] in ("long", "hedged"))
    short_exposure = sum(p["current_value"] for p in enriched if p["direction"] == "short")
    net_exposure   = long_exposure - short_exposure

    summary = {
        "total_positions":      len(enriched),
        "long_positions":       sum(1 for p in enriched if p["direction"] == "long"),
        "short_positions":      sum(1 for p in enriched if p["direction"] == "short"),
        "hedged_positions":     sum(1 for p in enriched if p["direction"] == "hedged"),
        "capital_at_entry":     round(total_capital_entry, 2),
        "current_value":        round(total_current_value, 2),
        "total_unrealized_pnl": round(total_unrealized, 2),
        "total_unrealized_pct": round((total_unrealized / total_capital_entry * 100) if total_capital_entry else 0, 2),
        "long_exposure":        round(long_exposure, 2),
        "short_exposure":       round(short_exposure, 2),
        "net_exposure":         round(net_exposure, 2),
    }

    # ----- Concentration -----
    # By symbol (largest first)
    by_symbol_raw = {}
    for p in enriched:
        by_symbol_raw[p["symbol"]] = by_symbol_raw.get(p["symbol"], 0) + p["current_value"]
    by_symbol = [
        {
            "symbol":        sym,
            "capital":       round(val, 2),
            "pct_of_book":   round((val / total_current_value * 100) if total_current_value else 0, 2),
            "warning":       _classify_concentration((val / total_current_value * 100) if total_current_value else 0, "symbol"),
        }
        for sym, val in sorted(by_symbol_raw.items(), key=lambda kv: kv[1], reverse=True)
    ]

    # By sector
    by_sector_raw = {}
    for p in enriched:
        sec = p["sector"] or "Unknown"
        by_sector_raw[sec] = by_sector_raw.get(sec, 0) + p["current_value"]
    by_sector = [
        {
            "sector":      sec,
            "capital":     round(val, 2),
            "pct_of_book": round((val / total_current_value * 100) if total_current_value else 0, 2),
            "warning":     _classify_concentration((val / total_current_value * 100) if total_current_value else 0, "sector"),
        }
        for sec, val in sorted(by_sector_raw.items(), key=lambda kv: kv[1], reverse=True)
    ]

    # Direction skew
    if total_current_value > 0:
        long_pct   = round(long_exposure  / total_current_value * 100, 2)
        short_pct  = round(short_exposure / total_current_value * 100, 2)
        hedged_val = sum(p["current_value"] for p in enriched if p["direction"] == "hedged")
        hedged_pct = round(hedged_val / total_current_value * 100, 2)
    else:
        long_pct = short_pct = hedged_pct = 0
    if long_pct >= 90:    skew = "long_heavy"
    elif short_pct >= 90: skew = "short_heavy"
    elif abs(long_pct - short_pct) <= 20: skew = "balanced"
    else: skew = "long_tilted" if long_pct > short_pct else "short_tilted"

    concentration = {
        "by_symbol":    by_symbol,
        "by_sector":    by_sector,
        "by_direction": {
            "long_pct":  long_pct,
            "short_pct": short_pct,
            "hedged_pct": hedged_pct,
            "skew":      skew,
        },
    }

    # ----- Stress test -----
    # For each NIFTY scenario, compute portfolio P&L impact.
    # long: position_change = value * beta * nifty_pct
    # short: position_change = -value * beta * nifty_pct
    # hedged: ~half delta exposure (long + protective put roughly = +0.5 delta)
    nifty_scenarios = [-10, -5, -2, 0, 2, 5]
    scenarios = []
    for nf_pct in nifty_scenarios:
        portfolio_change = 0
        for p in enriched:
            val  = p["current_value"]
            beta = p["beta"]
            d    = p["direction"]
            if d == "long":
                portfolio_change += val * beta * (nf_pct / 100)
            elif d == "short":
                portfolio_change -= val * beta * (nf_pct / 100)
            elif d == "hedged":
                portfolio_change += val * beta * (nf_pct / 100) * 0.5
        portfolio_change_pct = (portfolio_change / total_current_value * 100) if total_current_value else 0
        scenarios.append({
            "nifty_pct":            nf_pct,
            "portfolio_change":     round(portfolio_change, 2),
            "portfolio_change_pct": round(portfolio_change_pct, 2),
        })

    # ----- Warnings (cross-cutting) -----
    warnings_list = []
    # Direction skew warning
    if abs(long_pct - short_pct) > 80 and total_current_value > 0:
        warnings_list.append({
            "level":   "warning",
            "kind":    "direction_skew",
            "message": f"Book is heavily {'long' if long_pct > short_pct else 'short'}-tilted "
                       f"({max(long_pct, short_pct):.0f}% one-sided). Vulnerable to market direction.",
        })
    # Top symbol concentration
    if by_symbol and by_symbol[0]["warning"] in ("warning", "danger"):
        warnings_list.append({
            "level":   by_symbol[0]["warning"],
            "kind":    "symbol_concentration",
            "message": f"{by_symbol[0]['symbol']} is {by_symbol[0]['pct_of_book']:.0f}% of the book "
                       "— consider trimming or hedging.",
        })
    # Top sector concentration
    if by_sector and by_sector[0]["warning"] in ("warning", "danger"):
        warnings_list.append({
            "level":   by_sector[0]["warning"],
            "kind":    "sector_concentration",
            "message": f"{by_sector[0]['sector']} sector is {by_sector[0]['pct_of_book']:.0f}% "
                       "of the book — diversify or hedge sector risk.",
        })
    # Worst-case stress test severity
    worst = min((s["portfolio_change_pct"] for s in scenarios), default=0)
    if worst < -8:
        warnings_list.append({
            "level":   "warning",
            "kind":    "stress_severity",
            "message": f"Under -10% NIFTY scenario, book would drop {worst:.1f}%. Consider hedges.",
        })

    return jsonify({
        "as_of":         datetime.utcnow().isoformat() + "Z",
        "positions":     enriched,
        "summary":       summary,
        "concentration": concentration,
        "stress_test":   {"scenarios": scenarios},
        "warnings":      warnings_list,
    }), 200


# ---------------------------------------------------------------------------
# Hedge Suggestions — concrete hedging plays for the current book
# ---------------------------------------------------------------------------
# Reviews the open positions + risk metrics and suggests specific hedges
# across multiple categories:
#   - Same-stock options:  protective put, covered call, collar
#   - Same-stock futures:  short futures of a long position (or vice versa)
#   - Sector pair trade:   long X / short peer Y (via peer_map.py)
#   - Index hedge:         NIFTY/BANKNIFTY puts or short futures
#   - Diversification:     trim suggestions when over-concentrated
#
# Two-layer design:
#   1. Algorithmic layer identifies top 3-5 risks and maps each to
#      concrete hedge candidates with sizing + cost estimates.
#   2. AI layer (Claude Haiku) reviews the algo output, prioritizes,
#      writes rationale + trade-offs in plain language.
#
# Takes the risk-dashboard data as a POST body to avoid re-fetching
# Yahoo data (which is slow). Frontend reuses the already-loaded
# risk dashboard data.
# ---------------------------------------------------------------------------

def _build_algo_hedge_candidates(positions: list, summary: dict,
                                 concentration: dict, warnings: list) -> list[dict]:
    """
    Rule-based hedge candidates derived from the book.
    Each candidate is a structured dict that the AI layer will refine.
    """
    candidates = []
    by_symbol = concentration.get("by_symbol", []) or []
    by_sector = concentration.get("by_sector", []) or []
    by_direction = concentration.get("by_direction", {}) or {}
    long_pct  = by_direction.get("long_pct", 0)
    short_pct = by_direction.get("short_pct", 0)
    total_value = summary.get("current_value", 0) or 0

    # Lookup table: symbol -> position dict (for quick access)
    pos_map = {p["symbol"]: p for p in positions}

    # ---------- Category 1: Single-name concentration ----------
    # Top concentrated symbol gets BOTH options and futures hedge suggestions
    for s in by_symbol[:2]:   # top-2 concentrations
        if s["pct_of_book"] < 15:
            continue            # not concentrated enough to merit a hedge
        sym = s["symbol"]
        pos = pos_map.get(sym)
        if not pos:
            continue
        direction = pos["direction"]
        qty       = pos["qty"]
        price     = pos["current_price"]
        sector    = pos.get("sector", "Unknown")

        # 1a. Protective put / call (most natural same-stock hedge)
        if direction in ("long", "hedged"):
            # Long position: buy a protective put — strike ~5-7% OTM (ATM is expensive)
            target_strike = round(price * 0.95 / 10) * 10   # round to nearest 10
            candidates.append({
                "category":         "symbol_concentration",
                "risk_addressed":   f"{sym} is {s['pct_of_book']:.0f}% of book — single-name event risk",
                "instrument_type":  "protective_put",
                "instrument_label": f"Buy {sym} {target_strike} PE (next monthly expiry)",
                "direction":        "buy",
                "sizing_hint":      f"1 PE lot per ~{NSE_LOT_SIZES.get(sym, 0) or 1} shares "
                                    f"(your position = {qty} shares → ~{max(1, qty // (NSE_LOT_SIZES.get(sym, 0) or 1))} lot{'s' if qty // (NSE_LOT_SIZES.get(sym, 0) or 1) > 1 else ''})",
                "cost_estimate":    f"Premium typically 1-2% of underlying for a 5% OTM put. "
                                    f"For 1 lot, premium roughly ₹{int(price * (NSE_LOT_SIZES.get(sym, 0) or 1) * 0.015):,}–₹{int(price * (NSE_LOT_SIZES.get(sym, 0) or 1) * 0.025):,}",
                "max_loss_capped":  f"Below ₹{target_strike}, losses fully capped (minus premium)",
                "tradeoff":         "Costs premium upfront; bleeds theta if stock stays flat",
                "exit_signal":      f"Close hedge if {sym} decisively breaks above ₹{round(price * 1.05, 2)} (your concentration becomes a winner, not a risk)",
            })

            # 1b. Covered call (income-generating hedge — caps upside)
            call_strike = round(price * 1.05 / 10) * 10
            candidates.append({
                "category":         "symbol_concentration",
                "risk_addressed":   f"{sym} concentration with neutral-to-mild-upward outlook",
                "instrument_type":  "covered_call",
                "instrument_label": f"Sell {sym} {call_strike} CE (next monthly expiry)",
                "direction":        "sell",
                "sizing_hint":      f"1 CE lot per ~{NSE_LOT_SIZES.get(sym, 0) or 1} shares of underlying",
                "cost_estimate":    f"Premium received: roughly 0.8-1.5% of underlying "
                                    f"(~₹{int(price * (NSE_LOT_SIZES.get(sym, 0) or 1) * 0.01):,}–₹{int(price * (NSE_LOT_SIZES.get(sym, 0) or 1) * 0.015):,} per lot collected)",
                "max_loss_capped":  f"Not a downside hedge — generates income, caps upside above ₹{call_strike}",
                "tradeoff":         f"You lose participation above ₹{call_strike}. Best when you're bullish but not euphoric on the stock.",
                "exit_signal":      f"Close call if stock approaches strike (₹{call_strike}) and you still want upside",
            })

            # 1c. Collar (zero-cost hedge combining put buy + call sell)
            candidates.append({
                "category":         "symbol_concentration",
                "risk_addressed":   f"{sym} concentration — defined-range protection at minimal cost",
                "instrument_type":  "collar",
                "instrument_label": f"Buy {target_strike} PE + Sell {call_strike} CE (same expiry)",
                "direction":        "buy_put_sell_call",
                "sizing_hint":      f"Equal lots: 1 PE + 1 CE per ~{NSE_LOT_SIZES.get(sym, 0) or 1} shares",
                "cost_estimate":    "Near zero net premium (call premium offsets put premium). Some skew may leave a small credit or debit.",
                "max_loss_capped":  f"Below ₹{target_strike}: protected. Above ₹{call_strike}: capped.",
                "tradeoff":         f"Locks you in a {round((call_strike/target_strike - 1)*100)}% range. Gives up unlimited upside.",
                "exit_signal":      "Unwind if your view on the stock changes materially (either side)",
            })

            # 1d. Short futures (full hedge, no premium, but margin-heavy)
            candidates.append({
                "category":         "symbol_concentration",
                "risk_addressed":   f"{sym} concentration — neutralize directional exposure without paying premium",
                "instrument_type":  "short_futures",
                "instrument_label": f"Short {sym} futures (next monthly expiry)",
                "direction":        "sell",
                "sizing_hint":      f"Short {max(1, qty // (NSE_LOT_SIZES.get(sym, 0) or 1))} lot{'s' if qty // (NSE_LOT_SIZES.get(sym, 0) or 1) > 1 else ''} to fully neutralize",
                "cost_estimate":    "No premium. Margin requirement: typically ₹1-2L per lot (depends on stock + SPAN/exposure margin).",
                "max_loss_capped":  "Hedge is symmetric — fully offsets stock moves in either direction",
                "tradeoff":         "Locks in current value. Gives up ALL upside on the hedged portion. Requires margin (capital efficiency low).",
                "exit_signal":      "Cover futures when you want directional exposure back (typically after the risk catalyst passes — e.g., post-earnings)",
            })

        elif direction == "short":
            # Short position: protective call (analogous to put for longs)
            target_strike = round(price * 1.05 / 10) * 10
            candidates.append({
                "category":         "symbol_concentration",
                "risk_addressed":   f"Short {sym} ({s['pct_of_book']:.0f}% of book) — unlimited upside risk",
                "instrument_type":  "protective_call",
                "instrument_label": f"Buy {sym} {target_strike} CE (next monthly expiry)",
                "direction":        "buy",
                "sizing_hint":      f"1 CE lot per ~{NSE_LOT_SIZES.get(sym, 0) or 1} shares short",
                "cost_estimate":    f"Premium 1-2% of underlying (~₹{int(price * (NSE_LOT_SIZES.get(sym, 0) or 1) * 0.015):,} per lot)",
                "max_loss_capped":  f"Above ₹{target_strike}: short losses capped",
                "tradeoff":         "Costs premium; theta decay if stock stays flat",
                "exit_signal":      f"Close hedge if {sym} decisively breaks below ₹{round(price * 0.95, 2)}",
            })

    # ---------- Category 2: Sector concentration ----------
    # Top-1 sector if >35% gets BOTH index hedge and pair trade suggestions
    if by_sector and by_sector[0].get("pct_of_book", 0) >= 35:
        top_sector = by_sector[0]
        sector_name = top_sector["sector"]

        # 2a. Sector index hedge (BANKNIFTY for banks, NIFTY otherwise)
        # Indian retail: most sector hedging routes through BANKNIFTY for financials,
        # otherwise NIFTY is the only liquid index choice.
        index_choice = "BANKNIFTY" if "bank" in sector_name.lower() or "financial" in sector_name.lower() else "NIFTY"
        candidates.append({
            "category":         "sector_concentration",
            "risk_addressed":   f"{sector_name} is {top_sector['pct_of_book']:.0f}% of book — sector-wide shock risk",
            "instrument_type":  "index_put",
            "instrument_label": f"Buy {index_choice} ATM PE (next monthly expiry)",
            "direction":        "buy",
            "sizing_hint":      f"Size to cover ~₹{int(top_sector['capital']):,} of exposure. "
                                f"Each {index_choice} put lot covers ~₹{12_50_000 if index_choice == 'NIFTY' else 7_50_000:,} of notional at current levels (approximate).",
            "cost_estimate":    f"{index_choice} ATM monthly put: ~1-2% of notional (~₹15,000-30,000 per lot)",
            "max_loss_capped":  "Hedges sector beta against broad market drops",
            "tradeoff":         "Hedges INDEX risk, not stock-specific risk. Best when your sector concentration moves with the index.",
            "exit_signal":      f"Roll or unwind on the next monthly expiry, or close when {index_choice} bounces ≥5% off lows",
        })

        # 2b. Sector peer pair trade (long your name, short a peer in the same sector)
        # Find the concentrated names in this sector and use peer_map to suggest a short
        try:
            from peer_map import get_peers
        except ImportError:
            get_peers = None

        if get_peers:
            # Find a position in this sector to base the pair on
            sector_positions = [p for p in positions if p.get("sector") == sector_name and p["direction"] in ("long", "hedged")]
            if sector_positions:
                anchor = sorted(sector_positions, key=lambda p: p["current_value"], reverse=True)[0]
                try:
                    peers = get_peers(anchor["symbol"]) or []
                    # Filter to peers NOT already in book — don't suggest shorting what you're already long
                    held_symbols = set(p["symbol"] for p in positions)
                    peer_candidates = [p for p in peers if p not in held_symbols][:2]
                    if peer_candidates:
                        peer_str = " or ".join(peer_candidates)
                        candidates.append({
                            "category":         "sector_concentration",
                            "risk_addressed":   f"{sector_name} concentration — hedge sector beta while keeping {anchor['symbol']} alpha",
                            "instrument_type":  "pair_trade",
                            "instrument_label": f"Short {peer_str} futures (a peer of {anchor['symbol']})",
                            "direction":        "sell",
                            "sizing_hint":      f"Size short to ~₹{int(anchor['current_value'] * 0.5):,} (half your {anchor['symbol']} exposure) — keeps some directional bet",
                            "cost_estimate":    "Margin for short futures: ₹1-2L per lot. No premium decay.",
                            "max_loss_capped":  f"Reduces sector beta by ~50%. Stock-specific alpha (vs peer) stays exposed.",
                            "tradeoff":         f"You're betting {anchor['symbol']} outperforms {peer_str}. If both rally together, the short loses money but the long makes more.",
                            "exit_signal":      f"Cover the short if your view on relative strength changes, OR on expiry rollover",
                        })
                except Exception as e:
                    logger.debug("Peer lookup failed for hedge suggestion: %s", e)

    # ---------- Category 3: Direction skew ----------
    if abs(long_pct - short_pct) >= 70 and total_value > 0:
        # Heavily one-sided book → suggest a broad index hedge
        # For long-heavy: buy NIFTY puts. For short-heavy: buy NIFTY calls.
        if long_pct > short_pct:
            candidates.append({
                "category":         "direction_skew",
                "risk_addressed":   f"Book is {long_pct:.0f}% long — vulnerable to broad market drawdown",
                "instrument_type":  "index_put",
                "instrument_label": "Buy NIFTY ATM PE (next monthly expiry) — broad market hedge",
                "direction":        "buy",
                "sizing_hint":      f"Cover ~50% of long exposure (₹{int(summary['long_exposure'] * 0.5):,}). "
                                    f"NIFTY put lot covers ~₹{12_50_000:,} of notional.",
                "cost_estimate":    "ATM monthly NIFTY put: ~₹15,000-25,000 per lot",
                "max_loss_capped":  "Protects against systemic drawdowns; doesn't cover stock-specific drops",
                "tradeoff":         "Premium decays. Best held only when expecting a near-term broad correction.",
                "exit_signal":      "Close or roll on monthly expiry; or sell if NIFTY drops 5%+ (lock in profit on the hedge)",
            })

            # Alternative: short NIFTY futures (no premium, full delta hedge)
            candidates.append({
                "category":         "direction_skew",
                "risk_addressed":   f"Long-heavy book — neutralize market beta without paying premium",
                "instrument_type":  "index_short_futures",
                "instrument_label": "Short NIFTY futures (next monthly expiry)",
                "direction":        "sell",
                "sizing_hint":      f"Short to beta-weighted exposure (~₹{int(summary['long_exposure'] * 0.5):,}). "
                                    f"Each NIFTY lot = ~₹{12_50_000:,} of notional.",
                "cost_estimate":    "Margin: ₹1.2-1.5L per lot. No premium decay.",
                "max_loss_capped":  "Full delta hedge on the hedged portion. Symmetric.",
                "tradeoff":         "Gives up upside on hedged portion. Capital-heavy due to margin.",
                "exit_signal":      "Cover when you want directional exposure back, or on expiry rollover",
            })
        else:
            # Short-heavy book → hedge with NIFTY calls
            candidates.append({
                "category":         "direction_skew",
                "risk_addressed":   f"Book is {short_pct:.0f}% short — vulnerable to broad market rally / short squeeze",
                "instrument_type":  "index_call",
                "instrument_label": "Buy NIFTY ATM CE (next monthly expiry) — short-squeeze protection",
                "direction":        "buy",
                "sizing_hint":      f"Cover ~50% of short exposure (~₹{int(summary['short_exposure'] * 0.5):,}). "
                                    f"NIFTY call lot covers ~₹{12_50_000:,} of notional.",
                "cost_estimate":    "ATM monthly NIFTY call: ~₹15,000-25,000 per lot",
                "max_loss_capped":  "Protects against systemic rallies",
                "tradeoff":         "Premium decays. Use when you expect short-term up-move risk.",
                "exit_signal":      "Close on monthly expiry, or unwind if NIFTY rallies 5%+",
            })

    # ---------- Category 4: Stress severity (cross-cutting) ----------
    # If the -10% NIFTY stress shows >8% portfolio drop, suggest broader hedging
    # This is independent of direction skew — even a balanced book can be high-beta
    # (Note: this is a general nudge, the specific instruments above already cover it)

    # ---------- Category 5: Diversification (non-derivative hedge) ----------
    if by_symbol and by_symbol[0].get("pct_of_book", 0) >= 25:
        top_name = by_symbol[0]["symbol"]
        candidates.append({
            "category":         "diversification",
            "risk_addressed":   f"{top_name} ({by_symbol[0]['pct_of_book']:.0f}% of book) — consider reducing concentration directly",
            "instrument_type":  "trim_position",
            "instrument_label": f"Trim {top_name} by 30-40% and rotate capital into uncorrelated names",
            "direction":        "sell",
            "sizing_hint":      f"Sell ~30% of your {top_name} position. Deploy into low-correlation sectors (IT/Pharma if you're banks-heavy, etc.)",
            "cost_estimate":    "Brokerage + STT on the trim. No ongoing cost.",
            "max_loss_capped":  "Doesn't cap loss — but reduces single-name event risk permanently",
            "tradeoff":         "You give up some upside if the name keeps running. Buys you peace of mind and dry powder.",
            "exit_signal":      "N/A — this is a structural change, not a hedge to unwind",
        })

    return candidates


def _build_hedge_ai_prompt(positions: list, summary: dict, concentration: dict,
                           warnings: list, candidates: list) -> str:
    """Build the prompt for Claude to refine the algo candidates."""
    # Compact book snapshot
    by_symbol = concentration.get("by_symbol", []) or []
    by_sector = concentration.get("by_sector", []) or []
    by_direction = concentration.get("by_direction", {}) or {}

    book_lines = []
    for p in positions:
        book_lines.append(
            f"- {p['symbol']}: {p['direction']} {p['qty']} @ ₹{p['entry_price']} "
            f"(now ₹{p['current_price']}, value ₹{p['current_value']:,.0f}, "
            f"beta {p['beta']}, sector {p.get('sector', 'Unknown')})"
        )
    book_str = "\n".join(book_lines)

    cand_lines = []
    for i, c in enumerate(candidates, 1):
        cand_lines.append(
            f"{i}. [{c['category']}/{c['instrument_type']}] {c['instrument_label']}\n"
            f"   Risk: {c['risk_addressed']}\n"
            f"   Sizing: {c['sizing_hint']}\n"
            f"   Cost: {c['cost_estimate']}"
        )
    cand_str = "\n\n".join(cand_lines)

    skew  = by_direction.get("skew", "unknown")
    top_sym = by_symbol[0]["symbol"] if by_symbol else "—"
    top_sym_pct = by_symbol[0]["pct_of_book"] if by_symbol else 0
    top_sec = by_sector[0]["sector"] if by_sector else "—"
    top_sec_pct = by_sector[0]["pct_of_book"] if by_sector else 0

    return f"""You are a derivatives risk advisor for an Indian F&O retail trader. Review their book and the algorithmically-generated hedge candidates. Write a focused, practical hedge plan.

CURRENT BOOK:
{book_str}

SUMMARY:
- Total positions: {summary['total_positions']} ({summary['long_positions']} long, {summary['short_positions']} short, {summary['hedged_positions']} hedged)
- Capital at entry: ₹{summary['capital_at_entry']:,.0f}
- Current value (MTM): ₹{summary['current_value']:,.0f}
- Unrealized P&L: ₹{summary['total_unrealized_pnl']:,.0f} ({summary['total_unrealized_pct']:+.1f}%)
- Direction skew: {skew} (long {by_direction.get('long_pct', 0):.0f}% / short {by_direction.get('short_pct', 0):.0f}%)
- Top symbol concentration: {top_sym} ({top_sym_pct:.0f}%)
- Top sector concentration: {top_sec} ({top_sec_pct:.0f}%)

ALGORITHMIC HEDGE CANDIDATES:
{cand_str}

YOUR TASK:
1. Write a 2-3 paragraph book overview identifying the 2-3 PRIORITY risks (most pressing first). Be specific — name positions, percentages, sectors.
2. Rank the algorithmic candidates by priority (1 = most important to act on). Not all candidates need to be top-ranked — pick the 3-4 most impactful given THIS book.
3. For each candidate you rank, write a 1-2 sentence "why this hedge here" rationale that ties it to the specific risk in this book.
4. Conclude with 1 paragraph of "general advice" about overall hedge strategy (e.g., "Hedge only what you can't afford to lose" or "Don't over-hedge — the cost adds up").

Constraints:
- DO NOT invent strikes, premiums, or lot sizes — use only what's in the algorithmic candidates above.
- DO NOT recommend products not in the candidates (no exotic options, no global cross-asset).
- DO write in plain, direct English. Avoid jargon where simpler words work.
- DO NOT say "consult a financial advisor" — the user knows. Be concrete.
- Output MUST be valid JSON with this exact structure:
{{
  "narrative": "<2-3 paragraph book overview>",
  "rankings": [
    {{"candidate_index": <1-based index into algorithmic candidates>, "priority": <1-N>, "rationale": "<1-2 sentence why>"}},
    ...
  ],
  "general_advice": "<1 paragraph closing thoughts>"
}}

Return ONLY the JSON, no markdown fences, no preamble."""


@app.route("/api/hedge-suggestions", methods=["POST"])
@limiter.limit(AI_RATE_LIMIT)
def hedge_suggestions():
    """
    Generate hedge suggestions for the current book.
    Takes risk-dashboard data as POST body to avoid duplicate Yahoo fetches.
    """
    if not anthropic_client:
        return jsonify({"error": "AI service not configured"}), 500

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    positions     = body.get("positions") or []
    summary       = body.get("summary") or {}
    concentration = body.get("concentration") or {}
    warnings_list = body.get("warnings") or []

    if not positions:
        return jsonify({
            "narrative":      "No open positions to hedge. Once you have open trades, this section will analyze them and suggest specific hedging plays.",
            "hedges":         [],
            "general_advice": "Hedging starts with risk awareness. Use the Risk Dashboard daily; hedge only when concentration or skew warnings appear.",
        }), 200

    # ----- Algorithmic layer -----
    candidates = _build_algo_hedge_candidates(positions, summary, concentration, warnings_list)

    if not candidates:
        return jsonify({
            "narrative":      "Your current book doesn't show meaningful concentration, skew, or stress-risk warnings. No specific hedges are warranted right now.",
            "hedges":         [],
            "general_advice": "Continue monitoring. Set up alerts on the Risk Dashboard for when concentration crosses 20% in a single name or 35% in a single sector — those are the moments to hedge.",
        }), 200

    # ----- AI layer: refine and rank -----
    prompt = _build_hedge_ai_prompt(positions, summary, concentration, warnings_list, candidates)

    try:
        ai_resp = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = ai_resp.content[0].text.strip()
        # Strip code fences if Claude included them
        if ai_text.startswith("```"):
            ai_text = ai_text.split("```")[1]
            if ai_text.startswith("json"):
                ai_text = ai_text[4:]
            ai_text = ai_text.strip()
        ai_data = json.loads(ai_text)
    except Exception as e:
        logger.exception("Hedge AI call failed: %s", e)
        # Fall back: return algorithmic candidates without AI ranking
        return jsonify({
            "narrative":      "AI ranking unavailable — showing all algorithmic hedge candidates in default order. Review each based on which risk you find most pressing.",
            "hedges":         [{**c, "priority": i + 1, "rationale": ""} for i, c in enumerate(candidates)],
            "general_advice": "Pick at most 1-2 hedges to actually implement. Over-hedging burns capital on premiums and margin.",
            "ai_error":       str(e),
        }), 200

    # Merge AI rankings into the candidates
    rankings = {r["candidate_index"]: r for r in (ai_data.get("rankings") or [])}
    hedges = []
    for i, c in enumerate(candidates, 1):
        rank = rankings.get(i, {})
        hedges.append({
            **c,
            "priority":  rank.get("priority", 99),
            "rationale": rank.get("rationale", ""),
        })
    # Sort by priority asc (top-priority first); drop low-priority unranked items
    hedges = sorted(hedges, key=lambda h: h["priority"])

    return jsonify({
        "narrative":      ai_data.get("narrative", "").strip(),
        "hedges":         hedges,
        "general_advice": ai_data.get("general_advice", "").strip(),
        "model":          ANTHROPIC_MODEL,
    }), 200


# ---------------------------------------------------------------------------
# Kanban Trade Idea Pipeline
# ---------------------------------------------------------------------------
# Pre-trade idea tracking — 4 stages (watching → researching → setup_forming
# → setup_confirmed). Cards persist in their own table (kanban_ideas).
# When ready, a card is "promoted" to a journal entry (live trade) and the
# kanban card is deleted (clean handoff).
#
# Stale cards (no stage change in 30 days) auto-archive on every list call,
# keeping the active view focused on real opportunities.
# ---------------------------------------------------------------------------

@app.route("/api/kanban", methods=["GET"])
@limiter.limit(READ_RATE_LIMIT)
def list_kanban_ideas():
    """Return all non-archived kanban ideas, ordered by stage then position."""
    try:
        import kanban_store
    except Exception as e:
        logger.exception("Kanban list: kanban_store unavailable")
        return jsonify({"error": f"Kanban module unavailable: {e}"}), 500

    include_archived = request.args.get("include_archived", "").lower() == "true"
    try:
        ideas = kanban_store.list_ideas(include_archived=include_archived, limit=300)
    except Exception as e:
        logger.exception("Kanban list failed")
        return jsonify({"error": f"Failed to list ideas: {e}"}), 500

    # Group by stage for easier frontend rendering
    by_stage = {s: [] for s in ["watching", "researching", "setup_forming", "setup_confirmed"]}
    archived = []
    for idea in ideas:
        if idea.get("archived"):
            archived.append(idea)
        else:
            by_stage.setdefault(idea["stage"], []).append(idea)

    return jsonify({
        "by_stage": by_stage,
        "archived": archived if include_archived else [],
        "counts": {s: len(by_stage[s]) for s in by_stage},
    }), 200


@app.route("/api/kanban/<int:idea_id>", methods=["GET"])
@limiter.limit(READ_RATE_LIMIT)
def get_kanban_idea(idea_id):
    """Fetch a single kanban idea."""
    try:
        import kanban_store
    except Exception as e:
        return jsonify({"error": f"Kanban module unavailable: {e}"}), 500

    idea = kanban_store.get_idea(idea_id)
    if idea is None:
        return jsonify({"error": "Idea not found"}), 404
    return jsonify(idea), 200


@app.route("/api/kanban", methods=["POST"])
@limiter.limit(READ_RATE_LIMIT)
def create_kanban_idea():
    """Create a new kanban idea."""
    try:
        import kanban_store
    except Exception as e:
        return jsonify({"error": f"Kanban module unavailable: {e}"}), 500

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    try:
        idea = kanban_store.create_idea(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Kanban create failed")
        return jsonify({"error": f"Failed to create idea: {e}"}), 500

    return jsonify(idea), 201


@app.route("/api/kanban/<int:idea_id>", methods=["PATCH"])
@limiter.limit(READ_RATE_LIMIT)
def update_kanban_idea(idea_id):
    """Update a kanban idea (any field, including stage)."""
    try:
        import kanban_store
    except Exception as e:
        return jsonify({"error": f"Kanban module unavailable: {e}"}), 500

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    try:
        idea = kanban_store.update_idea(idea_id, payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Kanban update failed for id=%s", idea_id)
        return jsonify({"error": f"Failed to update idea: {e}"}), 500

    if idea is None:
        return jsonify({"error": "Idea not found"}), 404
    return jsonify(idea), 200


@app.route("/api/kanban/<int:idea_id>", methods=["DELETE"])
@limiter.limit(READ_RATE_LIMIT)
def delete_kanban_idea(idea_id):
    """Hard-delete a kanban idea."""
    try:
        import kanban_store
    except Exception as e:
        return jsonify({"error": f"Kanban module unavailable: {e}"}), 500

    try:
        deleted = kanban_store.delete_idea(idea_id)
    except Exception as e:
        logger.exception("Kanban delete failed for id=%s", idea_id)
        return jsonify({"error": f"Failed to delete idea: {e}"}), 500

    if not deleted:
        return jsonify({"error": "Idea not found"}), 404
    return jsonify({"ok": True, "id": idea_id}), 200


@app.route("/api/kanban/<int:idea_id>/promote", methods=["POST"])
@limiter.limit(READ_RATE_LIMIT)
def promote_kanban_idea(idea_id):
    """Promote a kanban idea to a live journal trade. Deletes the kanban card."""
    try:
        import kanban_store
    except Exception as e:
        return jsonify({"error": f"Kanban module unavailable: {e}"}), 500

    try:
        journal_row = kanban_store.promote_idea(idea_id)
    except ValueError as e:
        # Validation errors (missing fields) — return as 400
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Kanban promote failed for id=%s", idea_id)
        return jsonify({"error": f"Failed to promote idea: {e}"}), 500

    if journal_row is None:
        return jsonify({"error": "Idea not found"}), 404

    return jsonify({
        "ok": True,
        "journal_trade": journal_row,
        "message": f"Idea promoted to live trade #{journal_row.get('id')}",
    }), 200


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

    # Classify each item as MATERIAL or NOISE for trading decisions.
    # Rules-based first (fast/free); AI-backed only for ambiguous items.
    items = _classify_news_relevance(items, symbol)

    return items


# ---------------------------------------------------------------------------
# News relevance classifier
# ---------------------------------------------------------------------------
# Tags each news item with relevance ("material" | "noise") for trading.
# Two-pass strategy:
#   1. Rules-based keyword classifier handles ~70-80% of headlines instantly
#   2. Remaining ambiguous headlines go to a single batched AI call
# Each item also gets `relevance_source` ("rule" | "ai") for transparency.
# ---------------------------------------------------------------------------

# MATERIAL keywords — strongly suggest a tradable/actionable news item.
# Includes: financial results, corporate actions, regulatory, M&A,
# management changes, large stake transactions, dividends/buybacks.
_MATERIAL_PATTERNS = [
    # Financial results
    r"\b(q[1-4]|quarterly|annual)\s+(results?|earnings?)\b",
    r"\b(beats?|missed?|exceed(s|ed)?)\s+(estimates?|expectations?|consensus)\b",
    r"\b(profit|loss|revenue|ebitda|net income)\s+(rises?|falls?|grows?|drops?|jumps?|up|down|surges?)\b",
    r"\bguidance\s+(raised?|cut|revised?|lowered?)\b",
    # Corporate actions
    r"\b(buyback|bonus|stock split|share split|rights issue|dividend)\b",
    r"\b(merger|acquisition|takeover|m&a|stake (sale|purchase|buy))\b",
    r"\b(spin[\s-]?off|demerger|delisting|ipo)\b",
    # Regulatory / legal
    r"\b(rbi|sebi|cci|nclt|sat|cbi|ed|enforcement directorate)\b",
    r"\b(regulatory|approval|license|sanction|fine|penalty|probe|investigation|raid|fraud|lawsuit)\b",
    # Management / governance
    r"\b(ceo|cfo|md|managing director|chairman)\s+(resigns?|steps? down|appointed?|exits?|leaves?|joined?)\b",
    r"\b(resignation|appointment|reshuffle)\b",
    # Large transactions / ownership
    r"\b(block deal|bulk deal|promoter (buy|sell|stake)|insider)\b",
    r"\b(stake of \d+|acquir(es?|ed) \d+|sells? \d+%)\b",
    # Specific event types
    r"\b(order|contract)\s+(win|award|secured|received|worth)\b",
    r"\bcredit rating\s+(upgrade|downgrade|cut|raised?)" + r"[ds]?\b",
    r"\b(plant|factory)\s+(shut|closure|fire|accident|inaugurat)",
    r"\b(strike|lockout|union)\b",
]

# NOISE keywords — speculative, repetitive, or low-signal items.
_NOISE_PATTERNS = [
    # Analyst chatter (not the action itself)
    r"\b(target price|price target|broker(age)?\s+(call|view|report))\b",
    r"\b(buy|sell|hold|accumulate|reduce|outperform|underperform)\s+(rating|call|recommendation)\b",
    r"\banalyst[s]?\s+(say|predict|expect|see|believe|target)\b",
    # Generic market noise
    r"\b(stocks?\s+to\s+(watch|buy|sell)|top\s+(stock|pick)s?)\b",
    r"\b(rumor|speculation|reported(ly)?|may|could|might|likely to)\b",
    r"\b(intraday|today's call|chart|technical|support|resistance)\b",
    # Sector/index chatter without specific action
    r"\b(nifty|sensex|market|index)\s+(opens?|closes?|gains?|loses?|hits?)\b",
    r"\b(rally|correction|crash|surge|drop)\s+(in|on)\s+(nifty|sensex|market)\b",
]


def _rules_classify_news_item(title: str, source: str) -> str:
    """
    Returns "material", "noise", or "ambiguous" based on keyword patterns.
    Uses re.IGNORECASE. Material wins ties (rare but possible).
    """
    if not title:
        return "ambiguous"
    text = title.lower()

    # Check material first
    for pattern in _MATERIAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "material"
    # Check noise
    for pattern in _NOISE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "noise"
    return "ambiguous"


def _classify_news_relevance(items: list, symbol: str) -> list:
    """
    Tag each news item with `relevance` and `relevance_source`.
    First pass: rules-based (fast). Second pass: batch AI for ambiguous items.
    Failures are silent — items default to relevance="material" so users
    don't lose news to a misfiring classifier.
    """
    if not items:
        return items

    # Pass 1: rules-based
    ambiguous_indexes = []
    for i, item in enumerate(items):
        verdict = _rules_classify_news_item(item.get("title", ""), item.get("source", ""))
        if verdict == "ambiguous":
            ambiguous_indexes.append(i)
            item["relevance"] = "material"   # default; will be overwritten by AI if it runs
            item["relevance_source"] = "default"
        else:
            item["relevance"] = verdict
            item["relevance_source"] = "rule"

    # Pass 2: AI for ambiguous items (only if any exist and Anthropic client is available)
    if ambiguous_indexes and anthropic_client:
        ambiguous_titles = [
            f"{i + 1}. {items[idx]['title']}"
            for i, idx in enumerate(ambiguous_indexes)
        ]
        prompt = f"""Classify each headline as MATERIAL or NOISE for an Indian F&O retail trader.

MATERIAL = news that could affect the stock price meaningfully or change a trading thesis. Examples: earnings results, M&A, regulatory action, management changes, large transactions, contract wins, credit rating changes.

NOISE = analyst chatter, target price changes, generic stock-pick lists, broad market commentary, speculation without a concrete action.

Stock under analysis: {symbol}

Headlines to classify:
{chr(10).join(ambiguous_titles)}

Return ONLY a JSON array with one verdict per headline in order, like:
["material", "noise", "material", ...]

No preamble, no markdown, no explanations. Just the JSON array."""

        try:
            resp = anthropic_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            ai_text = resp.content[0].text.strip()
            if ai_text.startswith("```"):
                ai_text = ai_text.split("```")[1]
                if ai_text.startswith("json"):
                    ai_text = ai_text[4:]
                ai_text = ai_text.strip()
            verdicts = json.loads(ai_text)
            if isinstance(verdicts, list) and len(verdicts) == len(ambiguous_indexes):
                for i, idx in enumerate(ambiguous_indexes):
                    v = str(verdicts[i]).lower().strip()
                    if v in ("material", "noise"):
                        items[idx]["relevance"] = v
                        items[idx]["relevance_source"] = "ai"
            else:
                logger.warning("News AI classifier returned unexpected length: %d vs %d expected",
                               len(verdicts) if isinstance(verdicts, list) else -1,
                               len(ambiguous_indexes))
        except Exception as e:
            logger.warning("News AI classifier failed (non-fatal): %s", e)
            # Items keep their "default" relevance (material) — better to over-show
            # than to hide potentially-relevant news

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
@limiter.limit(AI_RATE_LIMIT)
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
@limiter.limit(AI_RATE_LIMIT)
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
@limiter.limit(AI_RATE_LIMIT)
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
@limiter.limit(AI_RATE_LIMIT)
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
@limiter.limit(AI_RATE_LIMIT)
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
            # Top 2 Call OI strikes (highest open interest = strongest resistance walls)
            # and top 2 Put OI strikes (strongest support walls). Sorted by OI desc.
            "top_call_oi_strikes": _top_n_oi_strikes(options_data, "call", n=2),
            "top_put_oi_strikes":  _top_n_oi_strikes(options_data, "put",  n=2),
        },
    }


def _top_n_oi_strikes(options_data, side: str, n: int = 2) -> list[dict]:
    """
    Extract the top-N strikes by Open Interest from the option chain.

    Args:
      options_data: OptionChainData object (may be None for non-F&O)
      side: "call" or "put" — which side's OI to rank by
      n: how many top strikes to return

    Returns:
      List of {strike: float, oi: int} dicts sorted by OI descending.
      Empty list if no options data or no strikes available.
    """
    if options_data is None or not options_data.top_strikes:
        return []
    oi_key = "call_oi" if side == "call" else "put_oi"
    # Sort all strikes by OI descending, filter out zero-OI entries
    sorted_strikes = sorted(
        [s for s in options_data.top_strikes if s.get(oi_key, 0) > 0],
        key=lambda s: s[oi_key],
        reverse=True,
    )
    return [
        {"strike": float(s["strike"]), "oi": int(s[oi_key])}
        for s in sorted_strikes[:n]
    ]


# ===========================================================================
# TRADE-PLAN RESPONSE CACHE (Postgres-backed, 10-min TTL, price-aware)
# ===========================================================================
# Caching strategy: when the user regenerates a plan for the same symbol/
# direction/size within 10 minutes AND the spot price hasn't moved more than
# ~0.25%, return the cached response instead of re-running options + AI.
#
# Key components:
#   - Symbol + direction + qty/capital + mode (new vs existing)
#   - Price bucket — log-scaled so each unit = ~0.25% price move (symbol-agnostic)
#   - For existing mode: also includes user's entry, stop, days_held
#
# A cache hit saves:
#   - ~5-10s of options data fetch
#   - 1 Anthropic API call (~₹0.10)
#   - Postgres bandwidth (small win)
#
# The response gets a `cached: true` flag so the frontend can show a badge.
# ===========================================================================

import math   # for log-scaled price bucketing

TRADE_PLAN_CACHE_TTL_SECONDS = 10 * 60   # 10 minutes


def _ensure_trade_plan_cache_schema():
    """Create the cache table if it doesn't exist. Idempotent."""
    try:
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_plan_cache (
                        cache_key   TEXT PRIMARY KEY,
                        payload     JSONB NOT NULL,
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS trade_plan_cache_created_at_idx
                    ON trade_plan_cache (created_at)
                """)
                c.commit()
    except Exception as e:
        logger.warning("Failed to ensure trade_plan_cache schema: %s", e)


def _compute_trade_plan_cache_key(
    symbol: str,
    direction: str,
    input_mode: str,
    qty: int,
    capital: float,
    mode: str,
    spot_price: float,
    existing_entry: Optional[float] = None,
    existing_stop: Optional[float] = None,
    days_held: Optional[int] = None,
) -> str:
    """
    Build a deterministic cache key from all inputs that affect the response.

    The price bucket uses log(price) * 400 to give symbol-agnostic 0.25%
    buckets — a 0.25% move in price changes the bucket id by 1 regardless
    of whether the stock costs ₹100 or ₹10,000.

    Hashed with SHA-256 so the key is a fixed-length string (cleaner in
    Postgres and avoids issues with special characters).
    """
    # Log-scaled bucket: ln(1.005) ≈ 0.005, so multiplying by 200 means each
    # 0.5%-wide bucket gets a unique integer id. The user picked "0.25%" caching,
    # which they described as "RELIANCE ₹1400 caches ₹1396.50 to ₹1403.50" —
    # i.e. ±0.25% from center = 0.5% total bucket width.
    # Symbol-agnostic: works the same for ₹50 stocks and ₹50,000 stocks.
    price_bucket = int(math.log(max(spot_price, 0.01)) * 200)

    # Size component: prefer qty if specified, else capital (rounded to nearest 1000)
    if input_mode == "qty":
        size_part = f"qty:{qty}"
    else:
        # Round capital to nearest ₹1000 so small typing variations still hit cache
        size_part = f"cap:{int(round(capital / 1000) * 1000)}"

    # Mode component: new mode is symbol+direction+size+bucket
    # Existing mode also includes the user's entry/stop/days_held
    if mode == "existing":
        existing_part = (
            f"|e_entry:{round(existing_entry or 0, 2)}"
            f"|e_stop:{round(existing_stop or 0, 2)}"
            f"|d_held:{days_held or 0}"
        )
    else:
        existing_part = ""

    raw_key = (
        f"tp|{symbol}|{direction}|{size_part}|{mode}|{price_bucket}"
        f"{existing_part}"
    )

    # Hash for fixed-length storage
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _get_cached_trade_plan(cache_key: str) -> Optional[dict]:
    """
    Look up a cached trade-plan response.

    Returns the payload + age in seconds if found and fresh (<TTL),
    or None if missing/stale. Stale entries are NOT auto-deleted here —
    they're cleaned by _purge_old_trade_plan_cache.
    """
    try:
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute("""
                    SELECT payload,
                           EXTRACT(EPOCH FROM (NOW() - created_at))::int AS age_seconds
                    FROM trade_plan_cache
                    WHERE cache_key = %s
                      AND created_at > NOW() - INTERVAL '%s seconds'
                """, (cache_key, TRADE_PLAN_CACHE_TTL_SECONDS))
                row = cur.fetchone()
                if row is None:
                    return None
                payload, age_seconds = row
                # payload is already JSON (jsonb) — psycopg2 returns it as dict
                return {"payload": payload, "age_seconds": int(age_seconds)}
    except psycopg2.errors.UndefinedTable:
        # Table doesn't exist yet — first run. Create it and return cache miss.
        _ensure_trade_plan_cache_schema()
        return None
    except Exception as e:
        # Cache failures should NEVER break the endpoint — just log and skip
        logger.warning("Trade-plan cache read failed for key %s: %s",
                       cache_key[:12], e)
        return None


def _save_cached_trade_plan(cache_key: str, payload: dict) -> None:
    """Save (or upsert) a trade-plan response to cache. Best-effort."""
    try:
        import json as _json
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute("""
                    INSERT INTO trade_plan_cache (cache_key, payload, created_at)
                    VALUES (%s, %s::jsonb, NOW())
                    ON CONFLICT (cache_key) DO UPDATE
                    SET payload = EXCLUDED.payload,
                        created_at = NOW()
                """, (cache_key, _json.dumps(payload)))
                c.commit()
    except psycopg2.errors.UndefinedTable:
        # Table doesn't exist yet — create and retry once
        _ensure_trade_plan_cache_schema()
        try:
            import json as _json
            with _conn() as c:
                with c.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trade_plan_cache (cache_key, payload, created_at)
                        VALUES (%s, %s::jsonb, NOW())
                        ON CONFLICT (cache_key) DO UPDATE
                        SET payload = EXCLUDED.payload,
                            created_at = NOW()
                    """, (cache_key, _json.dumps(payload)))
                    c.commit()
        except Exception as e:
            logger.warning("Trade-plan cache write failed (retry) for key %s: %s",
                           cache_key[:12], e)
    except Exception as e:
        logger.warning("Trade-plan cache write failed for key %s: %s",
                       cache_key[:12], e)


def _purge_old_trade_plan_cache() -> None:
    """
    Delete cache entries older than 1 hour. Called opportunistically.
    Keeps the table tiny (10-min entries don't accumulate forever).
    """
    try:
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute("""
                    DELETE FROM trade_plan_cache
                    WHERE created_at < NOW() - INTERVAL '1 hour'
                """)
                c.commit()
    except Exception:
        # Silent — purge is opportunistic
        pass


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


def _recompute_targets_from_entry(levels: dict, entry: float, direction: str) -> dict:
    """
    When user is in an existing position, their entry is fixed at `entry`.
    The algorithmic targets were computed from the entry-zone midpoint, so
    R-multiples need to be recomputed from the user's actual entry to be
    honest about reward potential.

    Stop loss is unchanged — it's still the algo-recommended stop.
    Add zones are also unchanged — they're forward-looking "if you wanted
    to add" prices, independent of where the user originally entered.
    """
    is_long = direction in ("long", "hedged")
    stop_price = levels["stop_loss"]["price"]
    risk_per_share = abs(entry - stop_price)
    if risk_per_share <= 0:
        return levels  # bail — avoid divide by zero

    # Recompute the risk_per_share on stop_loss so econ uses it correctly
    levels["stop_loss"]["risk_per_share"] = round(risk_per_share, 2)
    levels["stop_loss"]["distance_pct"] = round((stop_price - entry) / entry * 100, 2)

    # Recompute R-multiples for each target based on user's actual entry
    for tier_key in ("t1", "t2", "t3"):
        t = levels["targets"][tier_key]
        target_price = t["price"]
        distance = abs(target_price - entry)
        t["r_multiple"] = round(distance / risk_per_share, 2) if risk_per_share > 0 else 0
        t["pct_move"]   = round((target_price - entry) / entry * 100, 2)

    return levels


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
    mode: str = "new",
    existing_position: Optional[dict] = None,
    scoreboard: Optional[list[dict]] = None,
) -> str:
    """
    Build a focused AI prompt — algo already computed the numbers, AI just
    explains the rationale.

    For existing-position mode, prepends a context block with unrealized P&L,
    days held, and the user's stop vs algo stop comparison. Asks the AI to
    specifically advise on whether to:
      - Keep current stop or adopt the algo stop
      - Hold / partial exit / full exit at current spot
      - What conditions would trigger an early exit
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

    # ---- Existing position context block (only in "existing" mode) ----
    existing_block = ""
    if mode == "existing" and existing_position:
        ep = existing_position
        pnl_label   = "PROFIT" if ep["is_in_profit"] else "LOSS"
        pnl_color   = "📈" if ep["is_in_profit"] else "📉"
        tighter_lbl = "TIGHTER" if ep["algo_is_tighter"] else "LOOSER"

        existing_block = (
            f"\n## EXISTING POSITION (THIS IS A REPLAN, NOT A NEW ENTRY)\n\n"
            f"The trader ALREADY HOLDS this position. They are NOT entering — "
            f"they are asking how to manage the position they have.\n\n"
            f"- Position: {ep['qty']} shares · entered at ₹{ep['entry_price']} · "
            f"held for {ep['days_held']} days\n"
            f"- Current spot: ₹{p['current_price']:.2f}\n"
            f"- Unrealized {pnl_label}: {pnl_color} ₹{ep['unrealized_total']:,.0f} "
            f"({ep['unrealized_per_share']:+.2f}/share = {ep['unrealized_pct']:+.2f}%)\n\n"
            f"### TRADER'S CURRENT STOP vs ALGO'S RECOMMENDED STOP\n"
            f"- TRADER'S STOP: ₹{ep['current_stop']:.2f} "
            f"({ep['user_distance_pct']:+.2f}% from entry · loss if hit: ₹{ep['user_stop_loss_total']:,.0f})\n"
            f"- ALGO RECOMMENDS: ₹{ep['algo_stop_price']:.2f} "
            f"({ep['algo_distance_pct']:+.2f}% from entry · loss if hit: ₹{ep['algo_stop_loss_total']:,.0f})\n"
            f"- Algo's stop is {tighter_lbl} than the trader's current stop\n"
            f"- Algo's stop rationale: {ep['algo_stop_source']}\n"
        )

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

    # ---- Scoreboard block — the algorithm computed verdicts; AI fills reasons ----
    scoreboard_block = ""
    if scoreboard:
        verdict_emoji = {"favorable": "✓ FOR", "against": "✗ AGAINST", "neutral": "~ NEUTRAL"}
        rows_text = "\n".join(
            f"- {r['bucket']} · {r['factor']} = {r['value']} → {verdict_emoji.get(r['verdict'], r['verdict'])}"
            for r in scoreboard
        )
        scoreboard_block = (
            f"\n## SCOREBOARD (algorithm-computed verdicts for {direction_label})\n\n"
            f"Each factor has been algorithmically classified as FAVORABLE, "
            f"AGAINST, or NEUTRAL for the chosen direction. YOUR JOB is to "
            f"write a one-line reason for each row in the SCOREBOARD_REASONS "
            f"section at the END of your response (see instructions below).\n\n"
            f"{rows_text}\n"
        )

    # ---- Task framing differs significantly based on mode ----
    if mode == "existing":
        intro = (
            f"You are a senior trading desk strategist writing a POSITION MANAGEMENT memo "
            f"for a retail Indian F&O trader who ALREADY HOLDS a {direction_label} position "
            f"in {symbol}. They are NOT entering a new trade — they want to know how to "
            f"MANAGE the position they're sitting in."
        )
        task_framing = (
            "Your job is to advise on:\n"
            "  1. Whether to KEEP the current stop or ADOPT the algo-recommended stop\n"
            "  2. Whether to HOLD / TAKE PARTIAL PROFITS / EXIT FULLY at current spot\n"
            "  3. What conditions would force an EARLY EXIT before any target hits\n"
            "  4. Whether ADDING to the position makes sense from current spot\n"
            "Do NOT discuss entry timing — they're already in. Do NOT discuss whether "
            "the entry was a good idea — that's water under the bridge."
        )
    else:
        intro = (
            f"You are a senior trading desk strategist writing an execution memo for a "
            f"retail Indian F&O trader who has decided to take a {direction_label} "
            f"position in {symbol}. The numerical levels below have already been "
            f"computed algorithmically. Your job is NOT to second-guess them — they're "
            f"based on objective signals (ATR, OI strikes, SMAs, 52w levels). Your job "
            f"is to write the COMMENTARY that explains WHY each level makes sense and "
            f"WHAT TO WATCH FOR."
        )
        task_framing = ""

    # ---- Sections vary based on mode ----
    if mode == "existing":
        sections = """### Psychology Check
Open with 2-3 sentences specific to managing this EXISTING position:
- If in PROFIT: "Loss aversion will scream at you to take profit now. Don't unless T1 hit."
- If in LOSS: "Don't widen the stop to 'give it room' — that's how small losses become big ones."
- If held >10 days with no movement: "Opportunity cost is real. Set a time stop alongside the price stop."
Then 2 universal reminders relevant to managing an existing trade
(not entering one): journaling the deviation between plan vs execution,
avoiding revenge after a stop-out, etc.

### Stop Loss Recommendation — KEEP CURRENT OR ADOPT ALGO STOP?
This is the most important section. Pick ONE:
- **"Keep your current stop at ₹X"** — explain why their stop is the right one
- **"Adopt algo-recommended stop at ₹Y"** — explain why the algo's stop is better

Cite the specific signal (SMA, OI level, ATR distance) that drives your choice.
If user is in profit: also consider whether to TRAIL the stop closer to lock in gains.
If user is in loss: consider whether the algo's stop is even still ahead of current spot.

### Hold / Partial Exit / Full Exit
Given current spot vs T1/T2/T3 distances and the days held, recommend ONE:
- **HOLD** — stay in, let the targets work
- **PARTIAL EXIT** — book some now, hold the rest (specify what %)
- **FULL EXIT** — exit the entire position now
Justify with specific data: how far is current spot from T1? Is momentum still intact?

### Adding to Position (Should you?)
Given the user is already in at ₹{existing_entry if existing_position else '—'}, would adding
make sense from current spot? Reference the algo's add zones but interpret them in
light of the existing position's average price. If you'd add, specify the price level
and the position-size impact on blended average.

### Forward Targets (recalculated from your entry)
For each of T1/T2/T3:
- One sentence on what catalyst/condition would get price there
- The R-multiple shown is now measured from YOUR ACTUAL ENTRY, not the algo midpoint
- Specifically call out which target makes the most sense to scale out at

### Time Stop Considerations
Given the position has been held {existing_position['days_held'] if existing_position else 0} days:
- Is this trade on schedule? (compare days held to ATR-implied time to target)
- What's the "time stop" rule: if no progress in X more days, exit?
- For F&O traders: how does monthly expiry affect this decision?

### Watch List (3-4 bullets)
- Earnings dates / events that would force re-evaluation
- Price levels that should trigger immediate action (above or below)
- Volume + OI shifts that would change the picture
- Specific to {direction.upper()}: """ + (
            "Don't unwind the hedge early just because it's 'unused'. Roll only if put strikes drift far from spot." if direction == "hedged"
            else "Short-squeeze risk. Spot it via volume spike + gap up + supportive news." if direction == "short"
            else "Time decay: if 5 more trading days pass with no progress toward T1, time-stop the position."
        ) + """

### Important Caveats
- 2-3 honest risks that the algorithmic plan doesn't account for
- The trader's entry is FIXED — none of this advice changes that cost basis
- Acknowledge what's unknowable (gap risk, news shocks, etc.)"""
    else:
        sections = """### Psychology Check
This is the FIRST section because mental discipline matters more than perfect entries.
Write 2 parts:
1. **Setup-specific psychology (2-3 sentences)**: Identify the SPECIFIC emotional traps
   this """ + direction_label + """ setup will trigger for the trader. Examples to draw from:
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
- """ + (
            "If hedged: note any condition where the hedge becomes unnecessary" if direction == "hedged"
            else "If short: explicitly note short-squeeze risk and how to spot it" if direction == "short"
            else "Time decay: if the trade goes nowhere for N trading days, what's the rule?"
        ) + """

### Important Caveats
- 2-3 honest risks that the algorithmic plan doesn't account for
- Acknowledge what's unknowable (gap risk, news shocks, etc.)"""

    return f"""{intro}{(' ' + task_framing) if task_framing else ''}

## MARKET CONTEXT
- Current price: ₹{p['current_price']:.2f}
- ATR(14): ₹{levels['atr']:.2f}
- 30D change: {p['pct_change_1d']:+.2f}% (today) · 1D: {p['pct_change_1d']:+.2f}%
- RSI(14): {p['rsi_14']}
- SMA 50/200: ₹{p.get('sma_50', 0):.2f} / ₹{p.get('sma_200', 0):.2f}
- 52w range: ₹{p['low_52w']:.2f} – ₹{p['high_52w']:.2f}
{options_block}{analyst_block}{existing_block}{scoreboard_block}

## ALGORITHMIC TRADE PLAN (already computed)

### Entry
{'- Existing entry (FIXED): ₹' + f"{entry['mid']:.2f}" + ' — you are already in at this price' if mode == 'existing' else '- Patient entry zone: ₹' + f"{entry['low']:.2f}" + ' – ₹' + f"{entry['high']:.2f}" + chr(10) + '- Midpoint used for risk calc: ₹' + f"{entry['mid']:.2f}"}

### Stop loss
- Price: ₹{stop['price']:.2f} ({stop['distance_pct']:+.2f}% from {'YOUR entry' if mode == 'existing' else 'entry'})
- Rationale: {stop['source']}
- Risk per share: ₹{stop['risk_per_share']:.2f}

### Position adds (averaging / scaling)
- Add on adverse move: ₹{levels['add_zones']['adverse_1']:.2f}, ₹{levels['add_zones']['adverse_2']:.2f}
- Add on confirmation: ₹{levels['add_zones']['confirm_1']:.2f}, ₹{levels['add_zones']['confirm_2']:.2f}

### Profit targets (ladder, R-multiples measured from {'YOUR entry' if mode == 'existing' else 'entry midpoint'})
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

Write a focused {'POSITION MANAGEMENT' if mode == 'existing' else 'execution'} memo in markdown with these EXACT sections (no introduction,
no preamble — start straight with the first heading):

{sections}

## CRITICAL — FINAL SECTION: SCOREBOARD_REASONS

After all the prose sections above, you MUST include this final section EXACTLY
in this format (parseable by the system, NOT shown to the user):

## SCOREBOARD_REASONS
Factor name 1: one-line reason (≤ 18 words)
Factor name 2: one-line reason (≤ 18 words)
...

Rules for the reasons:
- Use the EXACT factor names from the SCOREBOARD section above
  (e.g. "RSI(14)", "Price vs SMA 50", "PCR (OI)", "Promoter holding")
- One line each, no markdown, no bullets, just "factor: reason"
- Be specific to the current value (don't write generic things like "this is bullish")
- For FAVORABLE rows: explain why this specific reading helps the {direction.upper()} thesis
- For AGAINST rows: explain why this specific reading hurts the {direction.upper()} thesis
- For NEUTRAL rows: explain why it's not a strong signal either way
- Keep each reason under 18 words — these render in a table cell

Tone: direct, no fluff, no hype, no "this is going to be great". Indian F&O trader,
serious about discipline. Use specific ₹ values when relevant. Maximum 900 words total."""


@app.route("/api/trade-plan", methods=["POST"])
@limiter.limit(AI_RATE_LIMIT)
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

        # ===== Existing position mode (optional) =====
        # When `mode == "existing"`, the API treats the trader as already
        # in the position and replans around their actual entry — showing
        # unrealized P&L, comparing their stop vs algo stop, computing
        # P&L at each target from their actual cost basis.
        "mode":            "new" | "existing",   # default: "new"
        "existing_entry":  2450.50,              # required if mode == "existing"
        "existing_stop":   2380.00,              # required if mode == "existing"
        "days_held":       12,                   # required if mode == "existing"
      }
    """
    if not anthropic_client:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    body = request.get_json(silent=True) or {}

    # --- Input validation: core fields ---
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

    # --- Input validation: existing-position fields (optional) ---
    mode = (body.get("mode") or "new").strip().lower()
    if mode not in ("new", "existing"):
        return jsonify({"error": "mode must be 'new' or 'existing'"}), 400

    existing_entry = None
    existing_stop  = None
    days_held      = None
    if mode == "existing":
        # Existing-position mode requires entry + stop + days_held; qty was
        # already collected above (it represents the user's current position size).
        if input_mode == "capital":
            return jsonify({
                "error": "existing_requires_qty",
                "message": "Existing position mode requires Quantity input (not capital).",
            }), 400
        try:
            existing_entry = float(body.get("existing_entry", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "existing_entry must be a number"}), 400
        if existing_entry <= 0:
            return jsonify({"error": "existing_entry must be > 0"}), 400

        try:
            existing_stop = float(body.get("existing_stop", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "existing_stop must be a number"}), 400
        if existing_stop <= 0:
            return jsonify({"error": "existing_stop must be > 0"}), 400

        # Validate stop is on the correct side of entry for the direction
        is_long = direction in ("long", "hedged")
        if is_long and existing_stop >= existing_entry:
            return jsonify({
                "error": "stop_wrong_side",
                "message": f"For LONG position, stop (₹{existing_stop}) must be BELOW entry (₹{existing_entry}).",
            }), 400
        if not is_long and existing_stop <= existing_entry:
            return jsonify({
                "error": "stop_wrong_side",
                "message": f"For SHORT position, stop (₹{existing_stop}) must be ABOVE entry (₹{existing_entry}).",
            }), 400

        try:
            days_held = int(body.get("days_held", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "days_held must be an integer"}), 400
        if days_held < 0:
            return jsonify({"error": "days_held must be >= 0"}), 400

    logger.info(
        "Trade plan for %s (%s, %s, %s=%s, days_held=%s)",
        symbol, direction, mode, input_mode, qty or capital, days_held,
    )

    # --- Fetch market data in parallel ---
    price_data = _fetch_price_data(symbol)
    if price_data is None:
        return jsonify({"error": f"Could not fetch price data for {symbol}"}), 404

    current_price = price_data["current_price"]

    # --- Check trade-plan cache BEFORE expensive fetches/AI call ---
    # We need the current price to bucket it for cache lookup. Price fetch is
    # cheap (~1-2s); options + ratings + AI is what we want to skip on hit.
    cache_key = _compute_trade_plan_cache_key(
        symbol=symbol,
        direction=direction,
        input_mode=input_mode,
        qty=qty,
        capital=capital,
        mode=mode,
        spot_price=current_price,
        existing_entry=existing_entry,
        existing_stop=existing_stop,
        days_held=days_held,
    )
    cached = _get_cached_trade_plan(cache_key)
    if cached:
        logger.info(
            "Trade-plan CACHE HIT for %s (%s, age=%ds, key=%s)",
            symbol, direction, cached["age_seconds"], cache_key[:12],
        )
        payload = cached["payload"]
        # Inject cache metadata so the frontend can show a "served from cache" tag
        payload["cached"] = True
        payload["cache_age_seconds"] = cached["age_seconds"]
        return jsonify(payload), 200

    # Cache miss — opportunistically purge old entries while we're here
    _purge_old_trade_plan_cache()

    # Parallel fetch: ratings, options, fundamentals (latter is cached 7 days)
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_ratings = executor.submit(_fetch_analyst_ratings, symbol)
        future_options = executor.submit(fetch_options_data, symbol, current_price)
        future_fundamentals = executor.submit(get_fundamentals, symbol) if FUNDAMENTALS_AVAILABLE else None
        ratings      = future_ratings.result()
        options_data = future_options.result()
        fundamentals = future_fundamentals.result() if future_fundamentals else None

    # Hedged direction requires options data — fail gracefully if unavailable
    if direction == "hedged" and not options_data:
        return jsonify({
            "error": "no_options_data",
            "message": (f"{symbol} is not in F&O segment — cannot hedge with a "
                        f"protective put. Choose Long or Short direction instead."),
        }), 400

    # --- Compute algorithmic levels (always done — used to recommend stop, targets) ---
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

    # ----- EXISTING POSITION MODE: rewrite "entry" to user's actual entry -----
    # This affects the targets/economics calculations downstream so P&L is
    # measured from the user's real cost basis, not the algorithmic midpoint.
    # We also stash the algo stop separately so the AI can compare both.
    algo_stop = levels["stop_loss"].copy()  # preserve original for comparison
    existing_position = None

    if mode == "existing":
        # Replace entry zone with the user's fixed entry
        levels["entry_zone"] = {
            "low":  round(existing_entry, 2),
            "mid":  round(existing_entry, 2),
            "high": round(existing_entry, 2),
            "is_fixed": True,
        }
        # Recompute targets relative to user's entry (so R-multiples are honest)
        levels = _recompute_targets_from_entry(levels, existing_entry, direction)

        # Build the existing-position context block
        is_long = direction in ("long", "hedged")
        unrealized_per_share = (current_price - existing_entry) if is_long else (existing_entry - current_price)
        unrealized_total     = unrealized_per_share * qty
        unrealized_pct       = (unrealized_per_share / existing_entry * 100)

        # User's stop in P&L terms
        user_stop_loss_per_share = abs(existing_entry - existing_stop)
        user_stop_loss_total     = user_stop_loss_per_share * qty

        # Algo's recommended stop in P&L terms (from user's actual entry)
        algo_stop_loss_per_share = abs(existing_entry - algo_stop["price"])
        algo_stop_loss_total     = algo_stop_loss_per_share * qty

        # Compare: is algo's stop tighter/looser than user's?
        if is_long:
            algo_is_tighter = algo_stop["price"] > existing_stop
        else:
            algo_is_tighter = algo_stop["price"] < existing_stop

        existing_position = {
            "entry_price":              round(existing_entry, 2),
            "current_stop":             round(existing_stop, 2),
            "days_held":                days_held,
            "qty":                      qty,
            "unrealized_per_share":     round(unrealized_per_share, 2),
            "unrealized_total":         round(unrealized_total, 2),
            "unrealized_pct":           round(unrealized_pct, 2),
            "is_in_profit":             unrealized_per_share > 0,
            "user_stop_loss_per_share": round(user_stop_loss_per_share, 2),
            "user_stop_loss_total":     round(user_stop_loss_total, 2),
            "algo_stop_price":          round(algo_stop["price"], 2),
            "algo_stop_source":         algo_stop["source"],
            "algo_stop_loss_per_share": round(algo_stop_loss_per_share, 2),
            "algo_stop_loss_total":     round(algo_stop_loss_total, 2),
            "algo_is_tighter":          algo_is_tighter,
            "algo_distance_pct":        round((algo_stop["price"] - existing_entry) / existing_entry * 100, 2),
            "user_distance_pct":        round((existing_stop - existing_entry) / existing_entry * 100, 2),
        }

        # Update levels.stop_loss to reflect the ALGO recommendation (the AI's
        # input). Frontend uses both `algo_stop` from existing_position and
        # `current_stop` to show side-by-side.

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
    # For existing mode, economics uses the user's actual entry (already
    # written into levels.entry_zone above) and the ALGO stop (the
    # forward-looking risk if user adopts the new stop).
    econ = _compute_position_economics(levels, qty, lot_size, hedge)

    # --- Build scoreboard (algorithmic verdicts; AI fills reasons) ---
    scoreboard = _build_scoreboard(direction, price_data, options_data, ratings, fundamentals)
    scoreboard_summary = _summarize_scoreboard(scoreboard)

    # --- AI commentary ---
    prompt = _build_trade_plan_prompt(
        symbol, direction, levels, econ, hedge,
        price_data, options_data, ratings,
        mode=mode, existing_position=existing_position,
        scoreboard=scoreboard,
    )

    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2800,    # bumped slightly for the scoreboard reasons
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
    except Exception as e:
        logger.exception("Anthropic API call failed: %s", e)
        return jsonify({"error": f"AI commentary failed: {str(e)}"}), 500

    # --- Parse AI scoreboard reasons from the commentary ---
    # AI returns a section like:
    #   ## SCOREBOARD_REASONS
    #   RSI(14): one-line reason
    #   PCR (OI): another one-line reason
    #   ...
    # We extract these and merge back into the scoreboard rows.
    # The reasons section is stripped from the visible commentary so the user
    # sees only the prose memo.
    scoreboard, ai_text = _parse_scoreboard_reasons(scoreboard, ai_text)

    # Build the response payload
    response_payload = {
        "symbol":              symbol,
        "analyzed_at":         datetime.utcnow().isoformat() + "Z",
        "direction":           direction,
        "input_mode":          input_mode,
        "mode":                mode,
        "lot_size":            lot_size,
        "levels":              levels,
        "economics":           econ,
        "hedge":               hedge,
        "existing_position":   existing_position,
        "scoreboard":          scoreboard,
        "scoreboard_summary":  scoreboard_summary,
        "ai_commentary":       ai_text,
        "model":               ANTHROPIC_MODEL,
        "cached":              False,   # this is a fresh response
    }

    # Save to cache for next time (10-min TTL, price-aware bucket).
    # Best-effort — never let cache write failures break the endpoint.
    _save_cached_trade_plan(cache_key, response_payload)

    return jsonify(response_payload), 200


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


# ===========================================================================
# SCOREBOARD — factor-by-factor verdict table for trade plan
# ===========================================================================
#
# Algorithmic verdict picker: for each factor (RSI, PCR, PE, etc.) we compute
# whether it's FAVORABLE, NEUTRAL, or AGAINST the user's direction (long/short).
#
# Each row gets a placeholder for an AI-written one-line reason that the
# AI prompt fills in. Backend computes the verdicts deterministically;
# AI does the prose.
#
# Returns a list of factor dicts:
#   {
#     "bucket":   "Trend & Momentum",
#     "factor":   "Price vs SMA50",
#     "value":    "+3.2%",
#     "verdict":  "favorable" | "neutral" | "against",
#     "reason":   ""   # to be filled by AI
#   }
# ===========================================================================

# Try to import fundamentals_fetcher — optional dependency. If the user
# hasn't added the module to this repo yet (it lives in the cron repo
# originally), the scoreboard will skip the fundamentals bucket gracefully.
try:
    from fundamentals_fetcher import get_fundamentals
    FUNDAMENTALS_AVAILABLE = True
except ImportError:
    logger.warning(
        "fundamentals_fetcher not available — scoreboard will skip "
        "fundamentals bucket. Copy fundamentals_fetcher.py from the "
        "cron repo to enable PE/ROE/growth/promoter signals."
    )
    FUNDAMENTALS_AVAILABLE = False
    def get_fundamentals(symbol):
        return None


def _verdict_for_long_short(value_is_bullish: Optional[bool], direction: str) -> str:
    """
    Convert a bullish/bearish signal to a verdict relative to user's direction.

    Args:
      value_is_bullish: True (bullish), False (bearish), None (neutral)
      direction: "long" | "short" | "hedged"

    Returns: "favorable" | "against" | "neutral"
    """
    if value_is_bullish is None:
        return "neutral"
    is_long_aligned = direction in ("long", "hedged")
    # If signal is bullish and user is long → favorable
    # If signal is bullish and user is short → against
    if value_is_bullish == is_long_aligned:
        return "favorable"
    return "against"


def _build_scoreboard(
    direction: str,
    price_data: dict,
    options_data: Optional["OptionChainData"],
    ratings: Optional[dict],
    fundamentals,  # Fundamentals dataclass or None
) -> list[dict]:
    """
    Build the factor-by-factor scoreboard.

    Each factor has a hardcoded rule for bullish/bearish/neutral based on
    common technical analysis conventions. Verdict is then mapped to
    FAVORABLE/AGAINST/NEUTRAL based on user's direction.

    Reasons are left empty — AI fills them later.
    """
    rows: list[dict] = []
    p = price_data
    price = p["current_price"]

    # ===== Bucket 1: Trend & Momentum =====
    sma_50 = p.get("sma_50") or 0
    sma_200 = p.get("sma_200") or 0

    # Price vs SMA50
    if sma_50 > 0:
        pct_from_sma50 = (price - sma_50) / sma_50 * 100
        is_bullish = pct_from_sma50 > 1.5 if pct_from_sma50 > 0 else (pct_from_sma50 > -1.5 if pct_from_sma50 < 0 else None)
        # Refined: clearly above (>1.5%) = bullish; clearly below (<-1.5%) = bearish; near = neutral
        if pct_from_sma50 > 1.5:
            is_bullish = True
        elif pct_from_sma50 < -1.5:
            is_bullish = False
        else:
            is_bullish = None
        rows.append({
            "bucket":  "Trend & Momentum",
            "factor":  "Price vs SMA 50",
            "value":   f"{pct_from_sma50:+.2f}%",
            "verdict": _verdict_for_long_short(is_bullish, direction),
        })

    # Price vs SMA200
    if sma_200 > 0:
        pct_from_sma200 = (price - sma_200) / sma_200 * 100
        if pct_from_sma200 > 2:
            is_bullish = True
        elif pct_from_sma200 < -2:
            is_bullish = False
        else:
            is_bullish = None
        rows.append({
            "bucket":  "Trend & Momentum",
            "factor":  "Price vs SMA 200",
            "value":   f"{pct_from_sma200:+.2f}%",
            "verdict": _verdict_for_long_short(is_bullish, direction),
        })

    # RSI(14)
    rsi_val = p.get("rsi_14") or 50
    if rsi_val >= 70:
        # Overbought — bearish for new longs, favorable for shorts
        is_bullish = False
    elif rsi_val <= 30:
        # Oversold — bullish for longs, bearish for shorts
        is_bullish = True
    elif 40 <= rsi_val <= 60:
        is_bullish = None
    elif rsi_val > 60:
        # Strong but not OB — bullish
        is_bullish = True
    else:
        # 30-40 — weak
        is_bullish = False
    rows.append({
        "bucket":  "Trend & Momentum",
        "factor":  "RSI(14)",
        "value":   f"{rsi_val:.1f}",
        "verdict": _verdict_for_long_short(is_bullish, direction),
    })

    # MACD signal
    macd_line = p.get("macd_line")
    macd_signal = p.get("macd_signal")
    if macd_line is not None and macd_signal is not None:
        macd_diff = macd_line - macd_signal
        if abs(macd_diff) < 0.05:
            is_bullish = None
        else:
            is_bullish = macd_diff > 0
        rows.append({
            "bucket":  "Trend & Momentum",
            "factor":  "MACD",
            "value":   f"{macd_diff:+.2f} ({'bullish cross' if macd_diff > 0 else 'bearish cross'})",
            "verdict": _verdict_for_long_short(is_bullish, direction),
        })

    # 30-day price change
    chg_30d = p.get("pct_change_30d") or 0
    if chg_30d > 5:
        is_bullish = True
    elif chg_30d < -5:
        is_bullish = False
    else:
        is_bullish = None
    rows.append({
        "bucket":  "Trend & Momentum",
        "factor":  "30-day move",
        "value":   f"{chg_30d:+.2f}%",
        "verdict": _verdict_for_long_short(is_bullish, direction),
    })

    # ===== Bucket 2: Technical Structure =====
    # Distance from 52w high/low
    high_52w = p.get("high_52w") or price
    low_52w = p.get("low_52w") or price
    pct_from_high = (price - high_52w) / high_52w * 100   # always <= 0
    pct_from_low  = (price - low_52w) / low_52w * 100     # always >= 0

    # For longs: near 52w high = bullish breakout potential BUT also resistance
    # We treat <5% from high as bullish (momentum), >20% from high as bearish (broken trend)
    if pct_from_high > -5:
        is_bullish = True   # near highs = strong
    elif pct_from_high < -25:
        is_bullish = False  # far from highs = weak trend
    else:
        is_bullish = None
    rows.append({
        "bucket":  "Technical Structure",
        "factor":  "Distance from 52w high",
        "value":   f"{pct_from_high:+.1f}%",
        "verdict": _verdict_for_long_short(is_bullish, direction),
    })

    # For longs: near 52w low = bearish; far from low = bullish (uptrend confirmed)
    if pct_from_low > 30:
        is_bullish = True    # well off lows = strong uptrend
    elif pct_from_low < 8:
        is_bullish = False   # near lows = weak
    else:
        is_bullish = None
    rows.append({
        "bucket":  "Technical Structure",
        "factor":  "Distance from 52w low",
        "value":   f"+{pct_from_low:.1f}%",
        "verdict": _verdict_for_long_short(is_bullish, direction),
    })

    # Bollinger Band position
    bb_upper = p.get("bb_upper")
    bb_lower = p.get("bb_lower")
    if bb_upper and bb_lower and bb_upper > bb_lower:
        bb_pct = (price - bb_lower) / (bb_upper - bb_lower) * 100
        if bb_pct > 80:
            # Near upper band — overextended for longs, oversold-pop opportunity for shorts (mean reversion)
            is_bullish = False
            bb_label = "near upper band"
        elif bb_pct < 20:
            is_bullish = True
            bb_label = "near lower band"
        else:
            is_bullish = None
            bb_label = "mid-band"
        rows.append({
            "bucket":  "Technical Structure",
            "factor":  "Bollinger position",
            "value":   f"{bb_pct:.0f}% ({bb_label})",
            "verdict": _verdict_for_long_short(is_bullish, direction),
        })

    # Volume vs 20-day average
    vol_ratio = p.get("volume_ratio") or 1.0
    if vol_ratio > 1.5:
        # High volume — confirmation; direction depends on price action
        # We treat high volume on an UP move as bullish, on a DOWN move as bearish
        is_bullish = chg_30d > 0  # rough proxy
    elif vol_ratio < 0.7:
        is_bullish = None   # weak volume = no conviction either way
    else:
        is_bullish = None
    rows.append({
        "bucket":  "Technical Structure",
        "factor":  "Volume vs 20d avg",
        "value":   f"{vol_ratio:.2f}x",
        "verdict": _verdict_for_long_short(is_bullish, direction),
    })

    # ===== Bucket 3: Options Sentiment =====
    if options_data is not None:
        # Composite signal — the main bullish/bearish read
        comp = (options_data.composite_signal or "").lower()
        if "strong bullish" in comp:
            is_bullish = True
        elif "bullish" in comp:
            is_bullish = True
        elif "strong bearish" in comp:
            is_bullish = False
        elif "bearish" in comp:
            is_bullish = False
        else:
            is_bullish = None
        rows.append({
            "bucket":  "Options Sentiment",
            "factor":  "Composite signal",
            "value":   f"{options_data.composite_signal} ({options_data.composite_strength}/5)",
            "verdict": _verdict_for_long_short(is_bullish, direction),
        })

        # PCR (calibrated for Indian stocks: >0.95 bullish, <0.55 bearish)
        pcr = options_data.pcr_oi
        if pcr >= 0.95:
            is_bullish = True
        elif pcr < 0.55:
            is_bullish = False
        else:
            is_bullish = None
        rows.append({
            "bucket":  "Options Sentiment",
            "factor":  "PCR (OI)",
            "value":   f"{pcr:.2f}",
            "verdict": _verdict_for_long_short(is_bullish, direction),
        })

        # Max Pain distance — only signals when stretched (>4%)
        if options_data.max_pain > 0:
            pct_from_pain = (price - options_data.max_pain) / options_data.max_pain * 100
            if pct_from_pain > 4:
                # Price above max pain — gravitational pull DOWN to expiry = bearish for new longs
                is_bullish = False
            elif pct_from_pain < -4:
                is_bullish = True
            else:
                is_bullish = None
            rows.append({
                "bucket":  "Options Sentiment",
                "factor":  "Max Pain distance",
                "value":   f"{pct_from_pain:+.1f}% ({'above' if pct_from_pain > 0 else 'below'} max pain ₹{options_data.max_pain:.0f})",
                "verdict": _verdict_for_long_short(is_bullish, direction),
            })

        # OI Buildup signal
        bld = (options_data.buildup_signal or "").lower()
        if bld == "long buildup":
            is_bullish = True
        elif bld == "short covering":
            is_bullish = True
        elif bld == "short buildup":
            is_bullish = False
        elif bld == "long unwinding":
            is_bullish = False
        else:
            is_bullish = None
        rows.append({
            "bucket":  "Options Sentiment",
            "factor":  "OI buildup",
            "value":   options_data.buildup_signal,
            "verdict": _verdict_for_long_short(is_bullish, direction),
        })

    # ===== Bucket 4: Analyst Sentiment =====
    if ratings and ratings.get("data_source") != "unavailable":
        # Consensus
        consensus = (ratings.get("consensus") or "").lower()
        if "strong buy" in consensus:
            is_bullish = True
        elif consensus == "buy":
            is_bullish = True
        elif "strong sell" in consensus:
            is_bullish = False
        elif consensus == "sell":
            is_bullish = False
        elif consensus == "hold" or consensus == "underperform" or consensus == "outperform":
            # Hold is neutral; under/outperform are weak signals
            if consensus == "outperform":
                is_bullish = True
            elif consensus == "underperform":
                is_bullish = False
            else:
                is_bullish = None
        else:
            is_bullish = None
        rows.append({
            "bucket":  "Analyst Sentiment",
            "factor":  "Consensus",
            "value":   ratings.get("consensus") or "—",
            "verdict": _verdict_for_long_short(is_bullish, direction),
        })

        # Mean target upside
        if ratings.get("price_target"):
            target = ratings["price_target"].get("mean")
            if target:
                upside_pct = (target - price) / price * 100
                if upside_pct > 8:
                    is_bullish = True
                elif upside_pct < -3:
                    is_bullish = False
                else:
                    is_bullish = None
                rows.append({
                    "bucket":  "Analyst Sentiment",
                    "factor":  "Mean target upside",
                    "value":   f"₹{target:.0f} ({upside_pct:+.1f}%)",
                    "verdict": _verdict_for_long_short(is_bullish, direction),
                })

        # Recent broker actions trend (last 8)
        actions = ratings.get("recent_actions") or []
        if actions:
            upgrades = sum(1 for a in actions if "up" in (a.get("action", "") or "").lower()
                           or "raised" in (a.get("action", "") or "").lower())
            downgrades = sum(1 for a in actions if "down" in (a.get("action", "") or "").lower()
                             or "cut" in (a.get("action", "") or "").lower())
            if upgrades > downgrades:
                is_bullish = True
            elif downgrades > upgrades:
                is_bullish = False
            else:
                is_bullish = None
            rows.append({
                "bucket":  "Analyst Sentiment",
                "factor":  "Recent broker actions",
                "value":   f"{upgrades} ↑ / {downgrades} ↓ (last {len(actions)})",
                "verdict": _verdict_for_long_short(is_bullish, direction),
            })

    # ===== Bucket 5: Fundamentals =====
    if fundamentals is not None:
        # PE ratio — for Indian stocks, broad benchmarks:
        # <15 = cheap (bullish for longs); 15-30 = fair; >50 = expensive (caution)
        if fundamentals.pe_ratio:
            pe = fundamentals.pe_ratio
            if pe < 0:
                # Negative PE = loss-making
                is_bullish = False
                pe_label = f"{pe:.1f} (loss-making)"
            elif pe < 18:
                is_bullish = True
                pe_label = f"{pe:.1f} (cheap)"
            elif pe > 50:
                is_bullish = False
                pe_label = f"{pe:.1f} (expensive)"
            else:
                is_bullish = None
                pe_label = f"{pe:.1f}"
            rows.append({
                "bucket":  "Fundamentals",
                "factor":  "PE ratio",
                "value":   pe_label,
                "verdict": _verdict_for_long_short(is_bullish, direction),
            })

        # ROE — for Indian stocks: >15% = good, <10% = poor
        if fundamentals.roe_pct:
            roe = fundamentals.roe_pct
            if roe > 18:
                is_bullish = True
            elif roe < 10:
                is_bullish = False
            else:
                is_bullish = None
            rows.append({
                "bucket":  "Fundamentals",
                "factor":  "ROE",
                "value":   f"{roe:.1f}%",
                "verdict": _verdict_for_long_short(is_bullish, direction),
            })

        # 3-year profit growth CAGR
        if fundamentals.profit_growth_3y_pct is not None:
            growth = fundamentals.profit_growth_3y_pct
            if growth > 15:
                is_bullish = True
            elif growth < 0:
                is_bullish = False
            else:
                is_bullish = None
            rows.append({
                "bucket":  "Fundamentals",
                "factor":  "3Y profit growth (CAGR)",
                "value":   f"{growth:+.1f}%",
                "verdict": _verdict_for_long_short(is_bullish, direction),
            })

        # Promoter holding — 35-75% = sweet spot
        if fundamentals.promoter_holding_pct:
            ph = fundamentals.promoter_holding_pct
            if 35 <= ph <= 75:
                is_bullish = True
            elif ph < 20:
                is_bullish = False
            else:
                is_bullish = None
            rows.append({
                "bucket":  "Fundamentals",
                "factor":  "Promoter holding",
                "value":   f"{ph:.1f}%",
                "verdict": _verdict_for_long_short(is_bullish, direction),
            })

    # Add empty reason placeholder for AI to fill
    for row in rows:
        row["reason"] = ""

    return rows


def _summarize_scoreboard(rows: list[dict]) -> dict:
    """Compute net counts for the summary line at the bottom."""
    favorable = sum(1 for r in rows if r["verdict"] == "favorable")
    neutral   = sum(1 for r in rows if r["verdict"] == "neutral")
    against   = sum(1 for r in rows if r["verdict"] == "against")
    net = favorable - against
    return {
        "favorable":  favorable,
        "neutral":    neutral,
        "against":    against,
        "total":      len(rows),
        "net_score":  net,
        # Verdict label based on net
        "verdict_label": (
            "Strongly Aligned" if net >= 6 else
            "Aligned"          if net >= 3 else
            "Mixed Signals"    if net >= -2 else
            "Misaligned"       if net >= -5 else
            "Strongly Misaligned"
        ),
    }


def _parse_scoreboard_reasons(scoreboard: list[dict], ai_text: str) -> tuple[list[dict], str]:
    """
    Extract the SCOREBOARD_REASONS section from the AI's response and merge
    the one-line reasons back into the scoreboard rows.

    Expected format from AI:
      ## SCOREBOARD_REASONS
      RSI(14): RSI 65 leaves room before overbought
      PCR (OI): 0.85 is neutral for Indian stocks
      ...

    Returns:
      (scoreboard with .reason fields populated, ai_text with section stripped)
    """
    import re

    # Find the section. Tolerant of various heading levels and capitalization.
    section_pattern = r"#{1,4}\s*SCOREBOARD[_\s]REASONS[\s\S]*?(?=\n#{1,4}\s|\Z)"
    match = re.search(section_pattern, ai_text, flags=re.IGNORECASE)

    if not match:
        # AI didn't include the section — fall back to leaving reasons blank
        logger.warning("AI response missing SCOREBOARD_REASONS section; reasons will be empty")
        return scoreboard, ai_text

    section_text = match.group(0)

    # Strip the heading line itself and parse the rest as "factor: reason"
    lines = section_text.split("\n")[1:]   # skip heading
    reason_map = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip leading markers like "- " or "* " from bullet lists
        line = re.sub(r"^[-*•]\s*", "", line)
        # Strip leading bold markers
        line = re.sub(r"^\*\*([^*]+)\*\*\s*[:\-]?\s*", r"\1: ", line)
        # Match "Factor name: reason text"
        m = re.match(r"^([^:]+?)\s*[:\-—]\s*(.+)$", line)
        if not m:
            continue
        factor_name = m.group(1).strip()
        reason = m.group(2).strip()
        # Use lower-cased factor name as key for fuzzy matching
        reason_map[factor_name.lower()] = reason

    # Merge reasons into scoreboard rows (match by factor name, case-insensitive)
    for row in scoreboard:
        factor_lower = row["factor"].lower()
        # Try exact match first, then partial
        if factor_lower in reason_map:
            row["reason"] = reason_map[factor_lower]
        else:
            # Find any key that's a substring match (handles minor naming drift)
            for key, val in reason_map.items():
                if key in factor_lower or factor_lower in key:
                    row["reason"] = val
                    break

    # Strip the section from ai_text so the user doesn't see the raw key:value list
    cleaned_text = re.sub(section_pattern, "", ai_text, flags=re.IGNORECASE).strip()
    return scoreboard, cleaned_text





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
@limiter.limit(AI_RATE_LIMIT)
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


# ===========================================================================
# TRADING JOURNAL — Postgres-backed CRUD for trade entries
# ===========================================================================
#
# Endpoints:
#   GET    /api/journal              — list all trades (optional ?status=open|closed)
#   GET    /api/journal/<id>         — fetch a single trade
#   POST   /api/journal              — create a new trade
#   PATCH  /api/journal/<id>         — partial update of an existing trade
#   POST   /api/journal/<id>/close   — close an open trade (sets exit price/date)
#   DELETE /api/journal/<id>         — permanently delete a trade
#
# All endpoints return JSON. Open trades get enriched with live spot price
# + unrealized P&L on list/get. Closed trades get realized P&L + R-multiple.
# ===========================================================================

import journal_store


def _current_spot_for_open(symbol: str) -> Optional[float]:
    """Try to fetch current spot price for an open position; return None on failure."""
    try:
        price_data = _fetch_price_data(symbol)
        if price_data:
            return float(price_data["current_price"])
    except Exception as e:
        logger.debug("Spot fetch failed for journal enrichment %s: %s", symbol, e)
    return None


@app.route("/api/journal", methods=["GET"])
@limiter.limit(READ_RATE_LIMIT)
def journal_list():
    """
    List trades. Query params:
      ?status=open    → only open
      ?status=closed  → only closed
      (no param)      → all
      ?limit=N        → cap results (default 100)
    """
    try:
        status = request.args.get("status")
        if status and status not in journal_store.VALID_STATUSES:
            return jsonify({"error": f"status must be one of {journal_store.VALID_STATUSES}"}), 400
        try:
            limit = int(request.args.get("limit", 100))
        except (TypeError, ValueError):
            return jsonify({"error": "limit must be an integer"}), 400
        limit = max(1, min(limit, 500))   # bounded

        # Run retention purge opportunistically (cheap — bounded by index)
        try:
            journal_store.purge_old_trades()
        except Exception as e:
            # Non-fatal — journal still works even if purge fails
            logger.warning("Journal purge failed: %s", e)

        raw = journal_store.list_trades(status=status, limit=limit)

        # Enrich each trade with derived metrics.
        # Open trades get a current-price lookup (best-effort, fail-soft).
        # We fetch each symbol's price only once even if multiple open
        # positions in same symbol.
        spot_cache: dict[str, Optional[float]] = {}
        enriched = []
        for trade in raw:
            current_price = None
            if trade["status"] == "open":
                sym = trade["symbol"]
                if sym not in spot_cache:
                    spot_cache[sym] = _current_spot_for_open(sym)
                current_price = spot_cache[sym]
            enriched.append(journal_store.enrich_with_metrics(trade, current_price))

        # Compute aggregate stats across closed trades for the dashboard widget
        closed_only = [t for t in enriched if t["status"] == "closed"]
        stats = journal_store.compute_aggregate_stats(closed_only)

        # Compute total unrealized P&L across open positions (rough portfolio gauge)
        open_only = [t for t in enriched if t["status"] == "open"]
        total_unrealized = sum(
            t.get("unrealized_pnl", 0) or 0 for t in open_only
        )

        return jsonify({
            "trades":             enriched,
            "stats":              stats,
            "total_unrealized":   round(total_unrealized, 2),
            "open_count":         len(open_only),
            "closed_count":       len(closed_only),
            "retention_days":     journal_store.RETENTION_DAYS,
            "fetched_at":         datetime.utcnow().isoformat() + "Z",
        }), 200
    except Exception as e:
        logger.exception("journal_list failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/journal/<int:trade_id>", methods=["GET"])
def journal_get(trade_id: int):
    """Fetch a single trade with enrichment."""
    try:
        trade = journal_store.get_trade(trade_id)
        if not trade:
            return jsonify({"error": "trade not found"}), 404

        current_price = None
        if trade["status"] == "open":
            current_price = _current_spot_for_open(trade["symbol"])
        enriched = journal_store.enrich_with_metrics(trade, current_price)
        return jsonify(enriched), 200
    except Exception as e:
        logger.exception("journal_get(%s) failed: %s", trade_id, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/journal", methods=["POST"])
@limiter.limit(READ_RATE_LIMIT)
def journal_create():
    """
    Create a new trade entry.

    Body (all but symbol/direction/qty/entry_price are optional):
      {
        "symbol":          "RELIANCE",
        "direction":       "long" | "short" | "hedged",
        "qty":             100,
        "entry_price":     2450.50,
        "entry_date":      "2026-05-15"   (default: today),
        "stop_loss":       2380.00,
        "target_price":    2600.00,
        "setup_type":      "breakout" | "reversal" | ...,
        "why_taken":       "free text..."
      }
    """
    try:
        body = request.get_json(silent=True) or {}
        created = journal_store.create_trade(body)
        return jsonify(created), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("journal_create failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/journal/<int:trade_id>", methods=["PATCH"])
def journal_update(trade_id: int):
    """Partial update of an existing trade. Body contains only fields to change."""
    try:
        body = request.get_json(silent=True) or {}
        updated = journal_store.update_trade(trade_id, body)
        if not updated:
            return jsonify({"error": "trade not found"}), 404
        return jsonify(updated), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("journal_update(%s) failed: %s", trade_id, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/journal/<int:trade_id>/close", methods=["POST"])
def journal_close(trade_id: int):
    """
    Close an open trade.

    Body:
      {
        "exit_price":      2580.00,         (required)
        "exit_date":       "2026-05-22",    (default: today)
        "went_well":       "free text...",
        "went_wrong":      "free text...",
        "emotional_state": "disciplined" | ...,
        "lesson_learned":  "free text..."
      }
    """
    try:
        body = request.get_json(silent=True) or {}

        try:
            exit_price = float(body.get("exit_price", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "exit_price must be a number"}), 400
        if exit_price <= 0:
            return jsonify({"error": "exit_price must be > 0"}), 400

        existing = journal_store.get_trade(trade_id)
        if not existing:
            return jsonify({"error": "trade not found"}), 404
        if existing["status"] == "closed":
            return jsonify({"error": "trade is already closed"}), 400

        closed = journal_store.close_trade(
            trade_id,
            exit_price      = exit_price,
            exit_date_str   = body.get("exit_date"),
            went_well       = body.get("went_well"),
            went_wrong      = body.get("went_wrong"),
            emotional_state = body.get("emotional_state"),
            lesson_learned  = body.get("lesson_learned"),
        )
        if not closed:
            return jsonify({"error": "close failed"}), 500
        return jsonify(closed), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("journal_close(%s) failed: %s", trade_id, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/journal/<int:trade_id>", methods=["DELETE"])
def journal_delete(trade_id: int):
    """Permanently delete a trade."""
    try:
        deleted = journal_store.delete_trade(trade_id)
        if not deleted:
            return jsonify({"error": "trade not found"}), 404
        return jsonify({"deleted": True, "id": trade_id}), 200
    except Exception as e:
        logger.exception("journal_delete(%s) failed: %s", trade_id, e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
