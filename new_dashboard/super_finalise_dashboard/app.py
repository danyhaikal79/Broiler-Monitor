"""
SUPER FINALISE DASHBOARD — login + role-gated access.

- Login screen (hardcoded demo accounts).
- admin : Monitor + Calibration
- staff : Monitor only

Run:
    python -m streamlit run new_dashboard/super_finalise_dashboard/app.py --server.address 0.0.0.0 --server.port 8500
Demo logins:  admin / admin123   ·   staff / staff123
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # add new_dashboard/ to path

import streamlit as st
from core import theme, auth, finalise_view, calibrate_view, history_view, alerts_view, control, config

st.set_page_config(page_title="Broiler Monitor · Secure", page_icon="🐔", layout="wide")
theme.inject_css()

# ---- gate ----
if not auth.is_logged_in():
    auth.login_screen()
    st.stop()

user = auth.current_user()
role = user["role"]

# ---- navigation (role-based) ----
# History + Alerts: both roles.  Calibration: admin only.
pages = ["Monitor", "History", "Alerts"] + (["Calibration"] if role == "admin" else [])

st.sidebar.markdown(
    f"<div style='padding:6px 2px 14px'>"
    f"<div style='font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:0.12em;"
    f"text-transform:uppercase;color:#7d8896'>signed in as</div>"
    f"<div style='font-weight:700;font-size:1.05rem;margin-top:2px'>{user['display']}</div>"
    f"<div style='font-family:JetBrains Mono;font-size:0.72rem;color:"
    f"{'#fbbf24' if role=='admin' else '#38bdf8'}'>{role.upper()}</div></div>",
    unsafe_allow_html=True,
)

choice = st.sidebar.radio("Navigate", pages, key="nav_page", label_visibility="collapsed")

if st.sidebar.button("⎋ Sign out", use_container_width=True):
    auth.logout()

st.sidebar.markdown("---")

# ---- worker state indicator (visible on EVERY page) ----
# Reflects the control flag the Monitor page sets + how fresh the worker's last
# publish is. The WORKER (separate process) is what logs; this is just a status hint.
_ctrl = control.read_control()
if _ctrl.get("camera_enabled"):
    _latest = control.read_latest()
    _interval = int(config.load().get("interval_sec", 30) or 30)
    _fresh = _latest is not None and control.is_fresh(_latest["age_s"], _interval)
    _color = "#a3e635" if _fresh else "#fbbf24"
    _sub = (f"updated {_latest['age_s']:.0f}s ago" if _latest else "waiting for worker…")
    if _latest is not None and not _fresh:
        _sub = f"stale ({_latest['age_s']:.0f}s) — worker running?"
    st.sidebar.markdown(
        f"<div style='padding:8px 10px;border:1px solid {_color};border-radius:8px;"
        f"background:rgba(163,230,53,0.06);font-family:JetBrains Mono;font-size:0.74rem;"
        f"margin-bottom:10px'>"
        f"<span style='color:{_color}'>● WORKER LIVE</span><br>"
        f"<span style='color:#7d8896'>{_sub}</span></div>",
        unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        "<div style='padding:8px 10px;border:1px solid #26303c;border-radius:8px;"
        "font-family:JetBrains Mono;font-size:0.74rem;margin-bottom:10px;color:#7d8896'>"
        "○ WORKER PAUSED<br><span>upload mode</span></div>",
        unsafe_allow_html=True)

# ---- render selected page ----
if choice == "Calibration" and role == "admin":
    calibrate_view.render(role=role)
elif choice == "History":
    history_view.render(role=role)
elif choice == "Alerts":
    alerts_view.render(role=role)
else:
    finalise_view.render(role=role)
