"""
Trading Journal — Postgres-backed CRUD for trade entries.

Schema: trading_journal
  - id              SERIAL PRIMARY KEY
  - symbol          TEXT NOT NULL
  - direction       TEXT NOT NULL              (long | short | hedged)
  - qty             INTEGER NOT NULL
  - entry_price     NUMERIC NOT NULL
  - entry_date      DATE NOT NULL
  - stop_loss       NUMERIC
  - target_price    NUMERIC                    (T1 — primary target)
  - setup_type      TEXT                       (breakout/reversal/momentum/...)
  - why_taken       TEXT                       (free-text reason)
  - status          TEXT NOT NULL DEFAULT 'open'   (open | closed)
  - exit_price      NUMERIC                    (filled on close)
  - exit_date       DATE                       (filled on close)
  - went_well       TEXT                       (post-trade review)
  - went_wrong      TEXT                       (post-trade review)
  - emotional_state TEXT                       (disciplined/fomo/panic/...)
  - lesson_learned  TEXT
  - created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
  - updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()

Retention policy (enforced by `purge_old_trades`):
  - CLOSED trades older than 30 days are auto-purged
  - OPEN trades are NEVER purged regardless of age (you can't lose a
    position you're still managing)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Retention: closed trades older than this many days are auto-purged.
# Open trades are NEVER purged — see purge_old_trades() for details.
RETENTION_DAYS = 30

# Valid enum values — kept here so both create and update validate consistently
VALID_DIRECTIONS = ("long", "short", "hedged")
VALID_STATUSES   = ("open", "closed")
VALID_SETUPS = (
    "breakout", "reversal", "momentum", "mean_reversion",
    "earnings", "news_based", "swing", "other",
)
VALID_EMOTIONS = (
    "disciplined", "fomo_exit", "panic_exit",
    "took_early", "held_too_long", "stopped_out", "other",
)


@contextmanager
def _conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    c = psycopg2.connect(DATABASE_URL)
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _ensure_schema() -> None:
    """Create table + index if missing. Idempotent — safe to call on every request."""
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trading_journal (
                    id              SERIAL PRIMARY KEY,
                    symbol          TEXT        NOT NULL,
                    direction       TEXT        NOT NULL,
                    qty             INTEGER     NOT NULL,
                    entry_price     NUMERIC     NOT NULL,
                    entry_date      DATE        NOT NULL,
                    stop_loss       NUMERIC,
                    target_price    NUMERIC,
                    setup_type      TEXT,
                    why_taken       TEXT,
                    status          TEXT        NOT NULL DEFAULT 'open',
                    exit_price      NUMERIC,
                    exit_date       DATE,
                    went_well       TEXT,
                    went_wrong      TEXT,
                    emotional_state TEXT,
                    lesson_learned  TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_journal_status_date
                ON trading_journal (status, entry_date DESC);
            """)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _normalize_payload(payload: dict, partial: bool = False) -> dict:
    """
    Normalize + validate incoming journal entry payload.

    Args:
      payload: raw dict from request body
      partial: if True, only validate the fields present (for UPDATE).
               If False, all required fields must be present (for CREATE).

    Raises ValueError on validation failure.
    """
    out = {}

    if "symbol" in payload or not partial:
        symbol = (payload.get("symbol") or "").strip().upper()
        if not symbol and not partial:
            raise ValueError("symbol is required")
        if symbol:
            if not all(c.isalnum() or c in "&-" for c in symbol):
                raise ValueError(f"invalid symbol: {symbol!r}")
            out["symbol"] = symbol

    if "direction" in payload or not partial:
        direction = (payload.get("direction") or "long").strip().lower()
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {VALID_DIRECTIONS}")
        out["direction"] = direction

    if "qty" in payload or not partial:
        try:
            qty = int(payload.get("qty", 0))
        except (TypeError, ValueError):
            raise ValueError("qty must be an integer")
        if qty <= 0:
            raise ValueError("qty must be > 0")
        out["qty"] = qty

    if "entry_price" in payload or not partial:
        try:
            entry_price = float(payload.get("entry_price", 0))
        except (TypeError, ValueError):
            raise ValueError("entry_price must be a number")
        if entry_price <= 0:
            raise ValueError("entry_price must be > 0")
        out["entry_price"] = entry_price

    if "entry_date" in payload or not partial:
        entry_date_str = (payload.get("entry_date") or "").strip()
        if not entry_date_str:
            if partial:
                pass  # leave it out
            else:
                # Default to today if missing on CREATE
                out["entry_date"] = date.today()
        else:
            try:
                out["entry_date"] = date.fromisoformat(entry_date_str)
            except ValueError:
                raise ValueError(f"entry_date must be YYYY-MM-DD format, got {entry_date_str!r}")

    # Optional fields
    if "stop_loss" in payload:
        v = payload.get("stop_loss")
        if v is None or v == "":
            out["stop_loss"] = None
        else:
            try:
                out["stop_loss"] = float(v)
            except (TypeError, ValueError):
                raise ValueError("stop_loss must be a number or null")

    if "target_price" in payload:
        v = payload.get("target_price")
        if v is None or v == "":
            out["target_price"] = None
        else:
            try:
                out["target_price"] = float(v)
            except (TypeError, ValueError):
                raise ValueError("target_price must be a number or null")

    if "setup_type" in payload:
        v = (payload.get("setup_type") or "").strip().lower() or None
        if v and v not in VALID_SETUPS:
            raise ValueError(f"setup_type must be one of {VALID_SETUPS}")
        out["setup_type"] = v

    for text_field in ("why_taken", "went_well", "went_wrong", "lesson_learned"):
        if text_field in payload:
            v = (payload.get(text_field) or "").strip() or None
            # Cap text length to 2000 chars to keep DB tidy
            if v and len(v) > 2000:
                v = v[:2000]
            out[text_field] = v

    if "status" in payload:
        status = (payload.get("status") or "open").strip().lower()
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}")
        out["status"] = status

    if "exit_price" in payload:
        v = payload.get("exit_price")
        if v is None or v == "":
            out["exit_price"] = None
        else:
            try:
                out["exit_price"] = float(v)
            except (TypeError, ValueError):
                raise ValueError("exit_price must be a number or null")

    if "exit_date" in payload:
        v = (payload.get("exit_date") or "").strip()
        if not v:
            out["exit_date"] = None
        else:
            try:
                out["exit_date"] = date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"exit_date must be YYYY-MM-DD format, got {v!r}")

    if "emotional_state" in payload:
        v = (payload.get("emotional_state") or "").strip().lower() or None
        if v and v not in VALID_EMOTIONS:
            raise ValueError(f"emotional_state must be one of {VALID_EMOTIONS}")
        out["emotional_state"] = v

    return out


def _row_to_dict(row: dict) -> dict:
    """Convert a Postgres row (RealDictCursor) to JSON-safe dict."""
    out = dict(row)
    # Date objects → ISO strings; Numeric → float
    for k, v in out.items():
        if isinstance(v, (date, datetime)):
            out[k] = v.isoformat()
        elif hasattr(v, "__float__") and k not in ("id", "qty"):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------
def list_trades(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    """
    List trades, optionally filtered by status (open/closed).
    Closed trades sorted by exit_date desc; open by entry_date desc.
    """
    _ensure_schema()

    if status and status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}")

    with _conn() as c:
        with c.cursor(cursor_factory=RealDictCursor) as cur:
            if status:
                cur.execute("""
                    SELECT * FROM trading_journal
                    WHERE status = %s
                    ORDER BY COALESCE(exit_date, entry_date) DESC, id DESC
                    LIMIT %s
                """, (status, limit))
            else:
                cur.execute("""
                    SELECT * FROM trading_journal
                    ORDER BY COALESCE(exit_date, entry_date) DESC, id DESC
                    LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
            return [_row_to_dict(r) for r in rows]


def get_trade(trade_id: int) -> Optional[dict]:
    """Fetch a single trade by id, or None if not found."""
    _ensure_schema()
    with _conn() as c:
        with c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM trading_journal WHERE id = %s", (trade_id,))
            row = cur.fetchone()
            return _row_to_dict(row) if row else None


def create_trade(payload: dict) -> dict:
    """Insert a new trade. Returns the created row."""
    _ensure_schema()
    data = _normalize_payload(payload, partial=False)

    # Default status to 'open' if not provided
    data.setdefault("status", "open")

    # Default entry_date if normalization didn't set it (shouldn't happen,
    # but be defensive)
    data.setdefault("entry_date", date.today())

    columns = list(data.keys())
    values  = [data[k] for k in columns]
    placeholders = ", ".join(["%s"] * len(columns))
    cols_sql     = ", ".join(columns)

    with _conn() as c:
        with c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"INSERT INTO trading_journal ({cols_sql}) VALUES ({placeholders}) RETURNING *",
                values,
            )
            row = cur.fetchone()
            logger.info("Journal: created trade #%s for %s", row["id"], row["symbol"])
            return _row_to_dict(row)


def update_trade(trade_id: int, payload: dict) -> Optional[dict]:
    """Update an existing trade. Returns the updated row, or None if id missing."""
    _ensure_schema()

    # Confirm trade exists first
    existing = get_trade(trade_id)
    if not existing:
        return None

    data = _normalize_payload(payload, partial=True)
    if not data:
        return existing  # nothing to update

    data["updated_at"] = datetime.now(timezone.utc)

    set_clause = ", ".join(f"{k} = %s" for k in data.keys())
    values = list(data.values()) + [trade_id]

    with _conn() as c:
        with c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"UPDATE trading_journal SET {set_clause} WHERE id = %s RETURNING *",
                values,
            )
            row = cur.fetchone()
            logger.info("Journal: updated trade #%s", trade_id)
            return _row_to_dict(row) if row else None


def close_trade(
    trade_id: int, exit_price: float, exit_date_str: Optional[str] = None,
    went_well: Optional[str] = None, went_wrong: Optional[str] = None,
    emotional_state: Optional[str] = None, lesson_learned: Optional[str] = None,
) -> Optional[dict]:
    """
    Close an open trade by setting status='closed' + exit_price + exit_date
    + optional post-trade review fields.
    """
    payload = {
        "status":          "closed",
        "exit_price":      exit_price,
        "exit_date":       exit_date_str or date.today().isoformat(),
        "went_well":       went_well,
        "went_wrong":      went_wrong,
        "emotional_state": emotional_state,
        "lesson_learned":  lesson_learned,
    }
    # Drop Nones from payload so the update only touches provided fields
    payload = {k: v for k, v in payload.items() if v is not None}

    return update_trade(trade_id, payload)


def delete_trade(trade_id: int) -> bool:
    """Permanently delete a trade. Returns True if deleted, False if id missing."""
    _ensure_schema()
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM trading_journal WHERE id = %s", (trade_id,))
            deleted = cur.rowcount > 0
            if deleted:
                logger.info("Journal: deleted trade #%s", trade_id)
            return deleted


def purge_old_trades() -> int:
    """
    Delete closed trades older than RETENTION_DAYS.
    Open trades are NEVER purged regardless of age (you can't lose a
    position you're still managing).

    Returns the number of trades deleted.
    """
    _ensure_schema()
    cutoff = date.today() - timedelta(days=RETENTION_DAYS)

    with _conn() as c:
        with c.cursor() as cur:
            cur.execute("""
                DELETE FROM trading_journal
                WHERE status = 'closed'
                  AND exit_date IS NOT NULL
                  AND exit_date < %s
            """, (cutoff,))
            count = cur.rowcount
            if count > 0:
                logger.info("Journal: purged %d trades closed before %s",
                            count, cutoff.isoformat())
            return count


# ---------------------------------------------------------------------------
# Derived metrics — added to each row server-side so frontend stays simple
# ---------------------------------------------------------------------------
def enrich_with_metrics(trade: dict, current_price: Optional[float] = None) -> dict:
    """
    Add derived/computed fields to a trade dict.

    For closed trades: realized P&L, R-multiple, % return, days held.
    For open trades: unrealized P&L (if current_price provided), days open.

    Returns a new dict — does not mutate the input.
    """
    t = dict(trade)
    is_long      = t["direction"] in ("long", "hedged")
    entry_price  = float(t["entry_price"])
    qty          = int(t["qty"])
    stop_loss    = float(t["stop_loss"]) if t.get("stop_loss") else None

    risk_per_share = abs(entry_price - stop_loss) if stop_loss else None

    # Days held (calendar days, not trading days — simpler + good enough)
    entry_dt = date.fromisoformat(t["entry_date"]) if isinstance(t["entry_date"], str) else t["entry_date"]
    if t["status"] == "closed" and t.get("exit_date"):
        exit_dt = date.fromisoformat(t["exit_date"]) if isinstance(t["exit_date"], str) else t["exit_date"]
        days_held = (exit_dt - entry_dt).days
    else:
        days_held = (date.today() - entry_dt).days

    t["days_held"] = days_held

    if t["status"] == "closed" and t.get("exit_price"):
        exit_price = float(t["exit_price"])
        pnl_per_share = (exit_price - entry_price) if is_long else (entry_price - exit_price)
        realized_pnl = pnl_per_share * qty
        pct_return = (pnl_per_share / entry_price * 100)
        t["realized_pnl"]       = round(realized_pnl, 2)
        t["realized_pnl_per_share"] = round(pnl_per_share, 2)
        t["pct_return"]         = round(pct_return, 2)
        t["is_winner"]          = realized_pnl > 0
        if risk_per_share and risk_per_share > 0:
            t["r_multiple"] = round(pnl_per_share / risk_per_share, 2)
        else:
            t["r_multiple"] = None
    elif t["status"] == "open" and current_price is not None:
        unrealized_per_share = (current_price - entry_price) if is_long else (entry_price - current_price)
        unrealized_pnl = unrealized_per_share * qty
        pct_unrealized = (unrealized_per_share / entry_price * 100)
        t["current_price"]            = round(current_price, 2)
        t["unrealized_pnl"]           = round(unrealized_pnl, 2)
        t["unrealized_pnl_per_share"] = round(unrealized_per_share, 2)
        t["unrealized_pct"]           = round(pct_unrealized, 2)
        t["is_in_profit"]             = unrealized_per_share > 0

    return t


def compute_aggregate_stats(closed_trades: list[dict]) -> dict:
    """
    Compute portfolio stats across a list of CLOSED trades.

    Returns: total trades, win rate, avg R-multiple, total P&L, best/worst,
    avg days held, breakdown by setup type.
    """
    if not closed_trades:
        return {
            "total_trades":  0,
            "win_rate":      None,
            "avg_r_multiple": None,
            "total_pnl":     0,
            "winners":       0,
            "losers":        0,
            "best_trade":    None,
            "worst_trade":   None,
            "avg_days_held": None,
            "by_setup":      {},
        }

    winners = [t for t in closed_trades if t.get("is_winner")]
    losers  = [t for t in closed_trades if t.get("is_winner") is False]

    r_multiples = [t["r_multiple"] for t in closed_trades if t.get("r_multiple") is not None]
    pnls        = [t.get("realized_pnl", 0) for t in closed_trades]
    days_held_list = [t.get("days_held") for t in closed_trades if t.get("days_held") is not None]

    by_setup = {}
    for t in closed_trades:
        setup = t.get("setup_type") or "untagged"
        if setup not in by_setup:
            by_setup[setup] = {"count": 0, "winners": 0, "total_pnl": 0}
        by_setup[setup]["count"] += 1
        if t.get("is_winner"):
            by_setup[setup]["winners"] += 1
        by_setup[setup]["total_pnl"] += t.get("realized_pnl", 0)

    # Best / worst trade summaries (avoid sending the whole row, keep it light)
    best = max(closed_trades, key=lambda t: t.get("realized_pnl", 0))
    worst = min(closed_trades, key=lambda t: t.get("realized_pnl", 0))

    return {
        "total_trades":   len(closed_trades),
        "winners":        len(winners),
        "losers":         len(losers),
        "win_rate":       round(len(winners) / len(closed_trades) * 100, 1) if closed_trades else 0,
        "avg_r_multiple": round(sum(r_multiples) / len(r_multiples), 2) if r_multiples else None,
        "total_pnl":      round(sum(pnls), 2),
        "best_trade":     {"symbol": best["symbol"], "pnl": best.get("realized_pnl", 0), "r": best.get("r_multiple")},
        "worst_trade":    {"symbol": worst["symbol"], "pnl": worst.get("realized_pnl", 0), "r": worst.get("r_multiple")},
        "avg_days_held":  round(sum(days_held_list) / len(days_held_list), 1) if days_held_list else None,
        "by_setup":       by_setup,
    }
