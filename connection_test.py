"""
Connectivity self-test — verify Supabase + Telegram are wired correctly,
WITHOUT needing the camera or the YOLO models.

It will:
  1. Show what monitor_config.json has configured.
  2. Insert a test reading + a test alert (Supabase cloud + local SQLite).
  3. Read them back.
  4. Send a test Telegram message.

Run:
  python connection_test.py

Then check:
  - Supabase  -> Table Editor -> 'readings' / 'alerts' should show the TEST rows.
  - Telegram  -> you should receive the test message in your bot chat.

This is just a diagnostic — safe to delete once everything works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "new_dashboard"))

from core import config, db, notify  # noqa: E402


def main():
    cfg = config.load()
    sb = config.is_supabase_configured(cfg)
    tg = config.is_telegram_configured(cfg)

    print("=" * 60)
    print("  BROILER MONITOR — connectivity self-test")
    print("=" * 60)
    print(f"  config file : {config.CONFIG_PATH}")
    print(f"               {'(found)' if config.CONFIG_PATH.exists() else '(NOT FOUND — copy monitor_config.example.json first)'}")
    print(f"  Supabase    : {'configured' if sb else 'NOT configured (url/key missing or still has YOUR-...)'}")
    print(f"  Telegram    : {'configured' if tg else 'NOT configured (token/chat_id missing or still has YOUR-...)'}")
    print(f"  local DB    : {db.LOCAL_DB}")
    print("-" * 60)

    # ---- 1. write a test reading + alert ----
    test_reading = {
        "feeder_type": "pan7kg", "feed_kg": 1.23, "fill_ratio": 0.15,
        "chicken_count": 42, "temperature_c": 30.0, "humidity_pct": 70.0,
        "thi_c": 28.5, "required_kg": 5.0, "feed_to_add_kg": 3.8, "coverage_pct": 24.6,
    }
    synced_r = db.log_reading(cfg, test_reading)
    synced_a = db.log_alert(cfg, "test", "connection_test.py self-test row", value=24.6)
    print(f"  [write] reading -> local OK · cloud {'OK' if synced_r else 'NOT synced'}")
    print(f"  [write] alert   -> local OK · cloud {'OK' if synced_a else 'NOT synced'}")
    if sb and not (synced_r and synced_a):
        print("          !! Supabase is configured but the cloud write FAILED.")
        print("             Check: URL correct? key correct? schema run (tables exist)?")
        print("             network reachable? RLS policies created (run supabase_schema.sql)?")

    # ---- 2. read back ----
    r = db.fetch_readings(cfg, limit=3)
    a = db.fetch_alerts(cfg, limit=3)
    src = "Supabase (cloud)" if sb else "local SQLite"
    print(f"  [read]  {len(r)} reading(s), {len(a)} alert(s) from {src}")
    if r:
        print(f"          newest reading: feed={r[0].get('feed_kg')}kg "
              f"cover={r[0].get('coverage_pct')}% @ {str(r[0].get('ts'))[:19]}")

    # ---- 3. telegram ----
    if tg:
        ok, info = notify.send_telegram(
            cfg["telegram_bot_token"], cfg["telegram_chat_id"],
            "✅ <b>Broiler Monitor</b> connection test — Telegram is working.")
        print(f"  [telegram] send -> {'SENT (check your chat)' if ok else 'FAILED'}  ({info})")
        if not ok:
            print("             Check: token correct? did you message the bot FIRST? chat_id correct?")
    else:
        print("  [telegram] skipped (not configured)")

    print("-" * 60)
    print("  Done. Now verify in Supabase Table Editor + your Telegram chat.")
    print("=" * 60)


if __name__ == "__main__":
    main()
