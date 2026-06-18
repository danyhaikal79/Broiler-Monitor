"""
History view — feed & flock trends logged by the background worker.

Reads from the database (Supabase cloud, local SQLite fallback) via core.db.
Shows the worker's most-recent reading as headline cards, then trend charts
and a recent-readings table. Does NOT touch the camera — the worker owns it.

Visible to both staff and admin.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from . import theme, config, db


def _fmt_age(ts_iso: str, offset_hours: float = 0.0) -> str:
    """Human 'x min ago'. Stored timestamps are LOCAL wall-clock (tz-naive), so we
    compare against local-now = UTC-now shifted by the same offset."""
    try:
        t = datetime.fromisoformat(str(ts_iso)[:19])      # naive local
        now_local = datetime.utcnow() + timedelta(hours=offset_hours)
        delta = now_local - t
        sec = int(delta.total_seconds())
        if sec < 0:
            return "just now"
        if sec < 90:
            return f"{sec}s ago"
        if sec < 5400:
            return f"{sec // 60} min ago"
        if sec < 172800:
            return f"{sec // 3600} h ago"
        return f"{sec // 86400} d ago"
    except Exception:
        return ts_iso[:16] if ts_iso else "—"


def _coverage_accent(pct: float, low_thr: float) -> str:
    if pct < low_thr:
        return theme.RED
    if pct < 50:
        return theme.AMBER
    return theme.LIME


def render(role: str | None = None, show_header: bool = True):
    if show_header:
        theme.header("MONITORING HISTORY", "feed & flock trends · logged by the worker", role)

    cfg = config.load()
    using_cloud = config.is_supabase_configured(cfg)
    low_thr = float(cfg.get("low_feed_coverage_pct", 25))
    off = float(cfg.get("display_tz_offset_hours", 0))   # used only for the 'x ago' math

    # ---- controls ----
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        limit = st.select_slider("Readings to load", options=[50, 100, 200, 500, 1000],
                                  value=200, key="hist_limit")
    with c2:
        live = st.checkbox("🔴 Auto-refresh (10s)", key="hist_live",
                           help="Re-read the database every 10 seconds.")
    with c3:
        st.caption(("☁ Source: **Supabase** (cloud) · falls back to local SQLite if offline"
                    if using_cloud else "💾 Source: **local SQLite** (Supabase not configured)"))

    if live:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=10000, key="hist_autorefresh")
        except ImportError:
            st.caption("⚠️ pip install streamlit-autorefresh for auto-refresh")
    if st.button("🔄 Refresh now", key="hist_refresh"):
        st.rerun()

    rows = db.fetch_readings(cfg, limit=int(limit))
    if not rows:
        st.info("No readings logged yet. Start the background worker on the device to begin "
                "logging:  `python3 worker.py`")
        with st.expander("ℹ️ Setup status"):
            st.write(f"- Supabase configured: **{'yes' if using_cloud else 'no'}**")
            st.write(f"- Telegram configured: **{'yes' if config.is_telegram_configured(cfg) else 'no'}**")
            st.write(f"- Local DB: `{db.LOCAL_DB}`")
        return

    # newest first from db; latest = rows[0]
    latest = rows[0]

    # ============================================================
    #  Latest reading (headline)
    # ============================================================
    theme.section("latest reading")
    cov = float(latest.get("coverage_pct") or 0.0)
    thi = float(latest.get("thi_c") or 0.0)
    heat = "heat stress" if thi >= 27 else ("warm" if thi >= 21 else "comfortable")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        theme.metric_card("Feed level", f"{latest.get('feed_kg', 0):.2f}", "kg",
                          str(latest.get("feeder_type") or "—"), accent=theme.LIME)
    with m2:
        theme.metric_card("Coverage", f"{cov:.0f}", "%", "of today's need",
                          accent=_coverage_accent(cov, low_thr))
    with m3:
        theme.metric_card("Chickens", f"{int(latest.get('chicken_count') or 0)}", "",
                          "detected", accent=theme.CYAN)
    with m4:
        theme.metric_card("THI", f"{thi:.1f}", "°C", heat,
                          accent=theme.RED if thi >= 27 else theme.AMBER)
    tzl = config.tz_label(cfg)
    st.caption(f"🕒 Last reading **{_fmt_age(latest.get('ts', ''), off)}** "
               f"· {config.fmt_local(latest.get('ts', ''), cfg)} {tzl} "
               f"· feed-to-add ~{float(latest.get('feed_to_add_kg') or 0):.2f} kg")

    # ============================================================
    #  Trends
    # ============================================================
    theme.section("trends over time")
    df = pd.DataFrame(rows)
    # Stored timestamps are already local wall-clock (tz-naive) — parse as-is so the
    # chart axis shows local time directly, no shift.
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    for col in ["feed_kg", "coverage_pct", "chicken_count", "thi_c", "temperature_c"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) < 2:
        st.caption("Need at least 2 readings to draw trend lines — charts appear as more data logs.")
    else:
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Feed level (kg)**")
            st.line_chart(df[["feed_kg"]], height=200, color=theme.LIME)
            st.markdown("**Flock size (chickens)**")
            st.line_chart(df[["chicken_count"]], height=200, color=theme.CYAN)
        with t2:
            st.markdown("**Coverage (% of daily need)**")
            st.line_chart(df[["coverage_pct"]], height=200, color=theme.AMBER)
            st.markdown("**Heat — THI & temperature (°C)**")
            heat_cols = [c for c in ["thi_c", "temperature_c"] if c in df.columns]
            st.line_chart(df[heat_cols], height=200)

    # ============================================================
    #  Recent readings table
    # ============================================================
    theme.section("recent readings")
    table = [{
        f"time ({tzl})": config.fmt_local(r.get("ts", ""), cfg),
        "feeder": r.get("feeder_type"),
        "feed_kg": r.get("feed_kg"),
        "fill_%": round(float(r.get("fill_ratio") or 0) * 100, 1),
        "birds": r.get("chicken_count"),
        "temp_°C": r.get("temperature_c"),
        "RH_%": r.get("humidity_pct"),
        "THI_°C": r.get("thi_c"),
        "need_kg": r.get("required_kg"),
        "add_kg": r.get("feed_to_add_kg"),
        "cover_%": r.get("coverage_pct"),
    } for r in rows[:200]]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(rows)} reading(s), newest first.")
