"""
NSE Dashboard API — read-only JSON endpoint for the Cloudflare Pages dashboard.

Exposes ONE endpoint: GET /api/snapshot
Returns the latest snapshots from all 3 report types in a single JSON blob.

Why a separate service rather than serving from the scanner repo? Cron jobs
are short-lived. A web service needs to be always-listening. Render's free
web service tier is perfect for this — and the cold start (~30s) is fine
because the cron jobs naturally warm it up.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager

import psycopg2
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Allow Cloudflare Pages to fetch this API from a different origin.
# Tighten to your specific Pages URL once you have it deployed.
CORS(app, origins=["*"])

DATABASE_URL = os.getenv("DATABASE_URL", "")


@contextmanager
def _conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    c = psycopg2.connect(DATABASE_URL)
    try:
        yield c
    finally:
        c.close()


@app.route("/")
def health():
    return jsonify({"status": "ok", "service": "nse-dashboard-api"}), 200


@app.route("/api/snapshot")
def get_snapshot():
    """
    Returns all latest snapshots in one call.

    Response shape:
      {
        "intraday": {"updated_at": "...", "items": [...], ...},
        "momentum": {"updated_at": "...", "items": [...], ...},
        "reversal": {"updated_at": "...", "items": [...], ...}
      }
    """
    try:
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute("""
                    SELECT report_type, updated_at, payload
                    FROM snapshots
                """)
                out = {}
                for report_type, updated_at, payload in cur.fetchall():
                    # payload is JSONB → psycopg2 returns it as a Python dict
                    out[report_type] = {
                        "updated_at": updated_at.isoformat(),
                        **payload,
                    }
                return jsonify(out), 200
    except psycopg2.errors.UndefinedTable:
        # Table doesn't exist yet — scanner hasn't run with the new code
        return jsonify({}), 200
    except Exception as e:
        logger.exception("Snapshot query failed: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
