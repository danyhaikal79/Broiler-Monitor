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
from core import theme, auth, finalise_view, calibrate_view

st.set_page_config(page_title="Broiler Monitor · Secure", page_icon="🐔", layout="wide")
theme.inject_css()

# ---- gate ----
if not auth.is_logged_in():
    auth.login_screen()
    st.stop()

user = auth.current_user()
role = user["role"]

# ---- navigation (role-based) ----
pages = ["Monitor"] + (["Calibration"] if role == "admin" else [])

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

# ---- render selected page ----
if choice == "Calibration" and role == "admin":
    calibrate_view.render(role=role)
else:
    finalise_view.render(role=role)
