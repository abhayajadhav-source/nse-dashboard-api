"""
Trade Idea Pipeline (Kanban) — Postgres-backed CRUD for pre-trade ideas.

Schema: kanban_ideas
  - id              SERIAL PRIMARY KEY
  - symbol          TEXT NOT NULL
  - stage           TEXT NOT NULL              (watching | researching | setup_forming | setup_confirmed)
  - direction       TEXT                       (long | short — optional in early stages)
  - thesis          TEXT                       (why interested in this name)
  - trigger_text    TEXT                       (what would move it to the next stage)
  - target_entry    NUMERIC                    (planned entry price)
  - target_stop     NUMERIC                    (planned stop price)
  - target_size     INTEGER                    (planned quantity)
  - notes           TEXT                       (free-form scratchpad)
  - position        INTEGER NOT NULL DEFAULT 0 (within-stage ordering, lower = top)
  - created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
  - updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
  - archived        BOOLEAN NOT NULL DEFAULT FALSE
  - archived_at     TIMESTAMPTZ
  - archived_reason TEXT                       ("stale" | "manual")

Pipeline philosophy:
  - Watching        — name caught my eye, no specific setup yet
  - Researching     — actively studying it (fundamentals, options chain, news)
  - Setup Forming   — technical/options pattern is developing, not actionable yet
  - Setup Confirmed — ready to execute; trigger is hit or imminent
  - (Live)          — handed off to trading_journal (kanban card is deleted)

Retention:
  - Cards that haven't changed stage in 30 days are auto-archived ("stale")
  - Archived cards are kept indefinitely but excluded from default list view

The schema name `trigger_text` (not `trigger`) avoids conflict with the
Postgres reserved keyword TRIGGER.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Cards untouched for this many days get auto-archived as "stale"
STALE_DAYS = 30

# Valid stages — order matters for display
STAGES = ["watching", "researching", "setup_forming", "setup_confirmed"]

# Valid directions (optional at early stages)
DIRECTIONS = ["long", "short", None]


@contextmanager
def _conn():
    """Yield a Postgres connection from DATABASE_URL. Caller controls commit."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def _ensure_schema() -> None:
    """Create the kanban_ideas table if it doesn't exist. Idempotent."""
    sql = """
    CREATE TABLE IF NOT EXISTS kanban_ideas (
        id              SERIAL PRIMARY KEY,
        symbol          TEXT NOT NULL,
        stage           TEXT NOT NULL,
        direction       TEXT,
        thesis          TEXT,
        trigger_text    TEXT,
        target_entry    NUMERIC,
        target_stop     NUMERIC,
        target_size     INTEGER,
        notes           TEXT,
        position        INTEGER NOT NULL DEFAULT 0,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        archived        BOOLEAN NOT NULL DEFAULT FALSE,
        archived_at     TIMESTAMPTZ,
        archived_reason TEXT
    );
    CREATE INDEX IF NOT EXISTS kanban_ideas_stage_idx
        ON kanban_ideas(stage, position) WHERE archived = FALSE;
    CREATE INDEX IF NOT EXISTS kanban_ideas_updated_idx
        ON kanban_ideas(updated_at);
    """
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute(sql)
            c.commit()


def _normalize_payload(payload: dict, partial: bool = False) -> dict:
    """
    Validate and normalize the payload for create/update.
    partial=True allows missing required fields (used by PATCH).

    Returns a dict suitable for direct SQL insertion (only valid columns).
    Raises ValueError on invalid inputs.
    """
    allowed = {
        "symbol", "stage", "direction", "thesis", "trigger_text",
        "target_entry", "target_stop", "target_size", "notes", "position",
        "archived", "archived_reason",
    }
    out = {k: v for k, v in payload.items() if k in allowed}

    # Required-field check (only on full creates)
    if not partial:
        if not out.get("symbol"):
            raise ValueError("symbol is required")
        if not out.get("stage"):
            raise ValueError("stage is required")

    # Symbol normalization (uppercase, strip)
    if "symbol" in out and out["symbol"] is not None:
        out["symbol"] = str(out["symbol"]).strip().upper()
        if not out["symbol"]:
            raise ValueError("symbol cannot be empty")

    # Stage validation
    if "stage" in out and out["stage"] is not None:
        if out["stage"] not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {out['stage']!r}")

    # Direction validation (allow None to clear)
    if "direction" in out and out["direction"] is not None:
        if out["direction"] not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {out['direction']!r}")

    # Numeric coercion
    for col in ("target_entry", "target_stop"):
        if col in out and out[col] is not None and out[col] != "":
            try:
                out[col] = float(out[col])
            except (TypeError, ValueError):
                raise ValueError(f"{col} must be numeric")
        elif col in out and (out[col] is None or out[col] == ""):
            out[col] = None

    if "target_size" in out and out["target_size"] is not None and out["target_size"] != "":
        try:
            out["target_size"] = int(out["target_size"])
            if out["target_size"] <= 0:
                raise ValueError("target_size must be positive")
        except (TypeError, ValueError) as e:
            if "positive" in str(e):
                raise
            raise ValueError("target_size must be an integer")
    elif "target_size" in out and (out["target_size"] is None or out["target_size"] == ""):
        out["target_size"] = None

    return out


def _row_to_dict(row: dict) -> dict:
    """Convert a Postgres row to a JSON-safe dict (timestamps, numerics)."""
    out = dict(row)
    for col in ("created_at", "updated_at", "archived_at"):
        if col in out and out[col] is not None:
            out[col] = out[col].isoformat() if hasattr(out[col], "isoformat") else str(out[col])
    # Cast numerics to floats / ints
    for col in ("target_entry", "target_stop"):
        if col in out and out[col] is not None:
            out[col] = float(out[col])
    for col in ("target_size", "position", "id"):
        if col in out and out[col] is not None:
            out[col] = int(out[col])
    return out


def list_ideas(include_archived: bool = False, limit: int = 200) -> list[dict]:
    """
    Return all kanban ideas, ordered by stage (in pipeline order)
    then by within-stage position, then by updated_at desc.
    """
    _ensure_schema()
    _archive_stale_ideas()  # Opportunistic — runs on every list call

    archived_clause = "" if include_archived else "WHERE archived = FALSE"
    # CASE expression orders stages in pipeline order
    sql = f"""
        SELECT *
        FROM kanban_ideas
        {archived_clause}
        ORDER BY
            CASE stage
                WHEN 'watching'         THEN 1
                WHEN 'researching'      THEN 2
                WHEN 'setup_forming'    THEN 3
                WHEN 'setup_confirmed'  THEN 4
                ELSE 5
            END,
            position ASC,
            updated_at DESC
        LIMIT %s
    """
    with _conn() as c:
        with c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def get_idea(idea_id: int) -> Optional[dict]:
    """Return a single idea or None if not found."""
    _ensure_schema()
    with _conn() as c:
        with c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM kanban_ideas WHERE id = %s", (idea_id,))
            row = cur.fetchone()
            return _row_to_dict(row) if row else None


def create_idea(payload: dict) -> dict:
    """Insert a new idea. Returns the created row as a dict."""
    _ensure_schema()
    data = _normalize_payload(payload, partial=False)

    # Default direction → None if not provided
    data.setdefault("direction", None)

    # Default position: bottom of the stage's current list
    if "position" not in data:
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 FROM kanban_ideas "
                    "WHERE stage = %s AND archived = FALSE",
                    (data["stage"],),
                )
                data["position"] = cur.fetchone()[0]

    columns = list(data.keys())
    placeholders = ", ".join(["%s"] * len(columns))
    values = [data[c] for c in columns]
    sql = (
        f"INSERT INTO kanban_ideas ({', '.join(columns)}) "
        f"VALUES ({placeholders}) RETURNING *"
    )
    with _conn() as c:
        with c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, values)
            row = cur.fetchone()
            c.commit()
    return _row_to_dict(row)


def update_idea(idea_id: int, payload: dict) -> Optional[dict]:
    """Patch-update an idea. Sets updated_at automatically. Returns updated row."""
    _ensure_schema()
    data = _normalize_payload(payload, partial=True)
    if not data:
        return get_idea(idea_id)

    # Build dynamic UPDATE
    set_clauses = [f"{col} = %s" for col in data.keys()]
    set_clauses.append("updated_at = NOW()")   # always bump updated_at
    values = list(data.values()) + [idea_id]
    sql = (
        f"UPDATE kanban_ideas SET {', '.join(set_clauses)} "
        f"WHERE id = %s RETURNING *"
    )
    with _conn() as c:
        with c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, values)
            row = cur.fetchone()
            c.commit()
    return _row_to_dict(row) if row else None


def delete_idea(idea_id: int) -> bool:
    """Hard delete an idea. Returns True if a row was deleted."""
    _ensure_schema()
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM kanban_ideas WHERE id = %s", (idea_id,))
            deleted = cur.rowcount > 0
            c.commit()
    return deleted


def _archive_stale_ideas() -> int:
    """
    Auto-archive ideas untouched for STALE_DAYS. Best-effort.
    Returns the number of rows archived.
    """
    try:
        with _conn() as c:
            with c.cursor() as cur:
                # STALE_DAYS is a module constant int — safe to inline.
                # psycopg2 doesn't bind parameters inside INTERVAL literals.
                cur.execute(
                    f"""
                    UPDATE kanban_ideas
                    SET archived = TRUE,
                        archived_at = NOW(),
                        archived_reason = 'stale'
                    WHERE archived = FALSE
                      AND updated_at < NOW() - INTERVAL '{int(STALE_DAYS)} days'
                    """
                )
                count = cur.rowcount
                c.commit()
                if count:
                    logger.info("Auto-archived %d stale kanban ideas", count)
                return count
    except Exception as e:
        logger.warning("Stale archive failed (non-fatal): %s", e)
        return 0


def promote_idea(idea_id: int) -> Optional[dict]:
    """
    Promote an idea to a live trade by creating a journal entry, then
    deleting the kanban card.

    Returns the new journal row, or None if the idea wasn't found.
    Requires the idea to have at least: direction, target_entry, target_size.
    """
    _ensure_schema()
    idea = get_idea(idea_id)
    if idea is None:
        return None

    # Validate required fields for promotion
    if not idea.get("direction"):
        raise ValueError("Cannot promote — direction is required (long/short)")
    if not idea.get("target_entry"):
        raise ValueError("Cannot promote — target_entry price is required")
    if not idea.get("target_size"):
        raise ValueError("Cannot promote — target_size (quantity) is required")

    # Build journal payload from the idea
    from datetime import date as _date
    journal_payload = {
        "symbol":      idea["symbol"],
        "direction":   idea["direction"],
        "qty":         idea["target_size"],
        "entry_price": idea["target_entry"],
        "entry_date":  _date.today().isoformat(),
        "stop_loss":   idea.get("target_stop"),
        "why_taken":   idea.get("thesis") or "Promoted from kanban pipeline",
        "setup_type":  "swing",   # safe default; user can edit later
        "status":      "open",
    }

    # Import locally to avoid circular import at module load
    import journal_store
    journal_row = journal_store.create_trade(journal_payload)

    # Delete the kanban card now that the trade is live
    delete_idea(idea_id)

    return journal_row


def list_archived(limit: int = 50) -> list[dict]:
    """Return recently archived ideas (for the 'recently archived' view)."""
    _ensure_schema()
    sql = """
        SELECT * FROM kanban_ideas
        WHERE archived = TRUE
        ORDER BY archived_at DESC
        LIMIT %s
    """
    with _conn() as c:
        with c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def unarchive_idea(idea_id: int) -> Optional[dict]:
    """Restore an archived idea (e.g., user recovers a stale one)."""
    return update_idea(idea_id, {
        "archived": False,
        "archived_reason": None,
    })
