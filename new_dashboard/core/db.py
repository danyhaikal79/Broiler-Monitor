"""
Readings + alerts storage.

Dual-write for reliability on a flaky network:
  - LOCAL SQLite  -> always written (never lost), file at <project>/monitor_local.sqlite
  - Supabase      -> best-effort cloud copy via its REST API (requests only; no SDK,
                     so it works on the Jetson's Python 3.8)

If the cloud write fails (offline), the row is still saved locally with synced=0,
so nothing is lost. Reads prefer Supabase (for remote access) and fall back to local.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import requests

from . import config

LOCAL_DB = config.ROOT / "monitor_local.sqlite"

READING_COLS = ["ts", "feeder_type", "feed_kg", "fill_ratio", "chicken_count",
                "temperature_c", "humidity_pct", "thi_c", "required_kg",
                "feed_to_add_kg", "coverage_pct"]
ALERT_COLS = ["ts", "kind", "message", "value"]


def now_iso(offset_hours: float = 0.0) -> str:
    """Timestamp string to store. We store LOCAL wall-clock time (UTC shifted by
    offset_hours, tz-naive) so the raw DB rows match the user's clock on a
    single-site deployment. Seconds precision, ISO 'YYYY-MM-DDTHH:MM:SS'."""
    return (datetime.now(timezone.utc) + timedelta(hours=offset_hours)).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- local SQLite
def _init_local():
    con = sqlite3.connect(LOCAL_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, feeder_type TEXT,
        feed_kg REAL, fill_ratio REAL, chicken_count INTEGER, temperature_c REAL,
        humidity_pct REAL, thi_c REAL, required_kg REAL, feed_to_add_kg REAL,
        coverage_pct REAL, synced INTEGER DEFAULT 0)""")
    con.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, message TEXT,
        value REAL, synced INTEGER DEFAULT 0)""")
    con.commit()
    return con


# ---------------------------------------------------------------- supabase REST
def _sb_headers(cfg):
    k = cfg["supabase_key"]
    return {"apikey": k, "Authorization": f"Bearer {k}",
            "Content-Type": "application/json", "Prefer": "return=minimal"}


def _sb_insert(cfg, table, row) -> bool:
    if not config.is_supabase_configured(cfg):
        return False
    try:
        r = requests.post(f"{cfg['supabase_url']}/rest/v1/{table}",
                          headers=_sb_headers(cfg), json=row, timeout=10)
        return r.status_code in (200, 201, 204)
    except Exception:
        return False


def _sb_select(cfg, table, limit=200):
    if not config.is_supabase_configured(cfg):
        return None
    try:
        r = requests.get(
            f"{cfg['supabase_url']}/rest/v1/{table}",
            headers={"apikey": cfg["supabase_key"], "Authorization": f"Bearer {cfg['supabase_key']}"},
            params={"select": "*", "order": "ts.desc", "limit": str(limit)},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- public API
def log_reading(cfg, reading: dict) -> bool:
    """Save one reading. Returns True if it reached the cloud (local always saved)."""
    reading = {**reading, "ts": reading.get("ts") or now_iso(cfg.get("display_tz_offset_hours", 0))}
    synced = _sb_insert(cfg, "readings", {k: reading.get(k) for k in READING_COLS})
    con = _init_local()
    con.execute(
        f"INSERT INTO readings ({','.join(READING_COLS)},synced) "
        f"VALUES ({','.join(['?']*len(READING_COLS))},?)",
        [reading.get(c) for c in READING_COLS] + [1 if synced else 0],
    )
    con.commit(); con.close()
    return synced


def log_alert(cfg, kind: str, message: str, value=None) -> bool:
    row = {"ts": now_iso(cfg.get("display_tz_offset_hours", 0)), "kind": kind, "message": message, "value": value}
    synced = _sb_insert(cfg, "alerts", row)
    con = _init_local()
    con.execute("INSERT INTO alerts (ts,kind,message,value,synced) VALUES (?,?,?,?,?)",
                [row["ts"], kind, message, value, 1 if synced else 0])
    con.commit(); con.close()
    return synced


def fetch_readings(cfg, limit=200) -> list[dict]:
    """Cloud first (remote access), local fallback. Newest first."""
    rows = _sb_select(cfg, "readings", limit)
    if rows is not None:
        return rows
    con = _init_local()
    cur = con.execute(f"SELECT {','.join(READING_COLS)} FROM readings ORDER BY ts DESC LIMIT ?", (limit,))
    out = [dict(zip(READING_COLS, r)) for r in cur.fetchall()]
    con.close()
    return out


def fetch_alerts(cfg, limit=100) -> list[dict]:
    rows = _sb_select(cfg, "alerts", limit)
    if rows is not None:
        return rows
    con = _init_local()
    cur = con.execute("SELECT ts,kind,message,value FROM alerts ORDER BY ts DESC LIMIT ?", (limit,))
    out = [dict(zip(["ts", "kind", "message", "value"], r)) for r in cur.fetchall()]
    con.close()
    return out


# ---------------------------------------------------------------- worker auto-discovery
# The worker (Jetson) upserts its CURRENT LAN ip:port here every ~10s; the dashboard reads
# it to locate the worker WITHOUT a hostname/mDNS (flaky for Python on Windows) or a
# hand-typed IP (changes on a hotspot). One row, name='worker'. Needs the worker_status
# table (deploy/supabase_worker_status.sql).
def publish_worker_status(cfg, ip: str, port: int) -> bool:
    """Worker -> Supabase: upsert this worker's reachable ip:port. Best-effort; never raises."""
    if not config.is_supabase_configured(cfg) or not ip:
        return False
    row = {"name": "worker", "ip": ip, "port": int(port),
           "updated_at": now_iso(cfg.get("display_tz_offset_hours", 0))}
    try:
        headers = dict(_sb_headers(cfg))
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"   # upsert on name
        r = requests.post(f"{cfg['supabase_url']}/rest/v1/worker_status",
                          headers=headers, json=row, timeout=10)
        return r.status_code in (200, 201, 204)
    except Exception:
        return False


def fetch_worker_status(cfg):
    """Dashboard <- Supabase: the worker's published {name, ip, port, updated_at}, or None."""
    if not config.is_supabase_configured(cfg):
        return None
    try:
        r = requests.get(
            f"{cfg['supabase_url']}/rest/v1/worker_status",
            headers={"apikey": cfg["supabase_key"], "Authorization": f"Bearer {cfg['supabase_key']}"},
            params={"select": "*", "name": "eq.worker", "limit": "1"}, timeout=8)
        if r.status_code == 200:
            rows = r.json()
            return rows[0] if rows else None
    except Exception:
        pass
    return None
