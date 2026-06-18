"""
Alerts view — low-feed and system events raised by the worker.

Reads the alerts log from the database (Supabase cloud, local SQLite fallback).
Admins also get a "send test Telegram" button to verify notifications are wired.

Visible to both staff and admin; the test-send button is admin-only.
"""

from __future__ import annotations

import streamlit as st

from . import theme, config, db, notify


def _chip(label: str, ok: bool) -> str:
    color = theme.LIME if ok else theme.MUTED
    state = "configured" if ok else "not set"
    return (f"<span style='display:inline-block;margin:0 8px 8px 0;padding:4px 10px;"
            f"border:1px solid {color};border-radius:999px;font-family:JetBrains Mono;"
            f"font-size:0.72rem;color:{color}'>{label}: {state}</span>")


_KIND_ICON = {"low_feed": "⚠️", "high_temp": "🔥", "test": "🧪", "error": "⛔", "info": "ℹ️"}


def render(role: str | None = None, show_header: bool = True):
    if show_header:
        theme.header("ALERTS LOG", "low-feed & system events", role)

    cfg = config.load()
    sb_ok = config.is_supabase_configured(cfg)
    tg_ok = config.is_telegram_configured(cfg)
    tzl = config.tz_label(cfg)

    # ---- integration status ----
    st.markdown(_chip("Supabase", sb_ok) + _chip("Telegram", tg_ok), unsafe_allow_html=True)

    cc1, cc2 = st.columns([3, 1])
    with cc2:
        if st.button("🔄 Refresh", key="alerts_refresh", use_container_width=True):
            st.rerun()

    alerts = db.fetch_alerts(cfg, limit=200)

    # ---- summary ----
    low_n = sum(1 for a in alerts if a.get("kind") == "low_feed")
    last_ts = config.fmt_local(alerts[0].get("ts", ""), cfg) if alerts else "—"
    s1, s2, s3 = st.columns(3)
    with s1:
        theme.metric_card("Total alerts", f"{len(alerts)}", "", "logged", accent=theme.CYAN)
    with s2:
        theme.metric_card("Low-feed alerts", f"{low_n}", "", "refill needed",
                          accent=theme.RED if low_n else theme.LIME)
    with s3:
        theme.metric_card("Most recent", last_ts or "—", "", tzl, accent=theme.AMBER)

    # ============================================================
    #  Admin: test the Telegram channel
    # ============================================================
    if role == "admin":
        theme.section("test notifications")
        if not tg_ok:
            st.info("Telegram isn't configured yet. Add `telegram_bot_token` and "
                    "`telegram_chat_id` to `monitor_config.json` to enable refill alerts.")
        else:
            if st.button("🧪 Send test Telegram alert", key="alerts_test"):
                ok, info = notify.send_telegram(
                    cfg["telegram_bot_token"], cfg["telegram_chat_id"],
                    "🧪 <b>Test alert</b>\nBroiler Monitor dashboard is wired up correctly.")
                db.log_alert(cfg, "test", "manual test alert from dashboard")
                if ok:
                    st.success("Test alert sent — check your Telegram chat.")
                else:
                    st.error(f"Could not send: {info}")

    # ============================================================
    #  Alert log
    # ============================================================
    theme.section("event log")
    if not alerts:
        st.info("No alerts yet. The worker raises a **low_feed** alert when coverage drops "
                f"below {float(cfg.get('low_feed_coverage_pct', 25)):.0f}% (with a "
                f"{float(cfg.get('alert_cooldown_min', 60)):.0f}-min cooldown so it doesn't spam).")
        return

    table = [{
        f"time ({tzl})": config.fmt_local(a.get("ts", ""), cfg),
        "": _KIND_ICON.get(a.get("kind"), "•"),
        "kind": a.get("kind"),
        "message": a.get("message"),
        "value": a.get("value"),
    } for a in alerts]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(alerts)} alert(s), newest first · "
               f"source: {'Supabase (cloud)' if sb_ok else 'local SQLite'}.")
