"""
Monitoring config loader — reads monitor_config.json from the project root.

Holds Supabase + Telegram credentials and the worker's settings (interval,
thresholds, feeder type, etc.). Secrets live in monitor_config.json which is
gitignored; monitor_config.example.json is the committed template.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "feeder_train").is_dir() and (p / "chicken_train").is_dir():
            return p
    return here.parent.parent


ROOT = _project_root()
CONFIG_PATH = ROOT / "monitor_config.json"

_DEFAULTS = {
    "supabase_url": "", "supabase_key": "",
    "telegram_bot_token": "", "telegram_chat_id": "",
    "camera_index": 0, "interval_sec": 600,
    # --- split deployment: dashboard (laptop) <-> worker (Jetson) over the LAN ---
    # The dashboard reaches the worker by its mDNS HOSTNAME (not its IP, which changes
    # on a hotspot). The worker serves /frame + /ping on worker_http_port.
    "worker_host": "auto",  # 'auto' = discover the worker's IP from Supabase (recommended). Or a hostname/IP, or 127.0.0.1.
    "worker_http_port": 8077,                 # worker's frame/heartbeat HTTP server
    "presence_timeout_sec": 25,               # worker logs on its own if no dashboard ping within this many seconds
    "dashboard_heartbeat_sec": 8,             # how often the (open) dashboard pings the worker, browser-side
    "live_refresh_sec": 5,                    # live-mode frame fetch + display cadence on the dashboard
    "feeder_method": 1,  # 1 = whole feeder, 2 = open area, 3 = demo/dummy farm
    "display_tz_offset_hours": 8,    # store UTC, DISPLAY local; Malaysia (MYT) = +8
    "display_tz_label": "MYT",
    "feeder_type": "pan7kg", "flock_start_date": None,
    "chicken_age_days_fallback": 21, "chicken_count_override": None,
    "low_feed_coverage_pct": 25, "alert_cooldown_min": 60,
    "high_temp_alert_c": 38.0,   # heat-emergency alert threshold (DVS guide: >38°C = survival zone)
    "use_sensor": True, "serial_port": "/dev/ttyUSB0", "serial_baud": 115200,
    "temperature_fallback_c": 25.0, "humidity_fallback_pct": 70.0,
}


def load() -> dict:
    cfg = dict(_DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in data.items() if not k.startswith("_")})
        except Exception as e:
            print(f"[config] could not parse {CONFIG_PATH}: {e}")
    return cfg


def chicken_age_days(cfg: dict) -> int:
    """Age from flock_start_date if set (auto-increments daily), else fallback."""
    fsd = cfg.get("flock_start_date")
    if fsd:
        try:
            start = datetime.strptime(str(fsd), "%Y-%m-%d").date()
            return max(0, (date.today() - start).days)
        except Exception:
            pass
    return int(cfg.get("chicken_age_days_fallback", 21))


def is_supabase_configured(cfg: dict) -> bool:
    return bool(cfg.get("supabase_url") and cfg.get("supabase_key")
                and "YOUR-" not in cfg["supabase_url"])


def is_telegram_configured(cfg: dict) -> bool:
    return bool(cfg.get("telegram_bot_token") and cfg.get("telegram_chat_id")
                and "YOUR-" not in str(cfg["telegram_bot_token"]))


def tz_label(cfg: dict) -> str:
    """Short timezone label for display, e.g. 'MYT' or 'UTC+8'."""
    lbl = cfg.get("display_tz_label")
    if lbl:
        return str(lbl)
    off = float(cfg.get("display_tz_offset_hours", 0))
    return f"UTC{off:+g}"


def fmt_local(ts_iso, cfg: dict = None) -> str:
    """Tidy a stored timestamp for display: 'YYYY-MM-DD HH:MM:SS'.

    Timestamps are STORED in local time (see db.now_iso), so this only formats
    the string — it does NOT shift the timezone.
    """
    return str(ts_iso)[:19].replace("T", " ")
