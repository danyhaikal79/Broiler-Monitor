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
import streamlit.components.v1 as components
from core import theme, auth, finalise_view, calibrate_view, history_view, alerts_view, worker_client, config

st.set_page_config(page_title="Broiler Monitor · Secure", page_icon="🐔", layout="wide")
theme.inject_css()


@st.cache_data(ttl=6, show_spinner=False)
def _worker_online(url: str) -> bool:
    """Cached so we ping the worker for the badge at most once every 6s (not per rerun)."""
    return worker_client.ping_url(url)

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

# ---- worker connectivity indicator + presence heartbeat (visible on EVERY page) ----
# Split deployment: the worker runs on the Jetson. We (a) show whether it's reachable
# over the LAN, and (b) keep pinging it from the BROWSER so it knows the dashboard is
# open and stands down (lets the laptop do the inference + logging). The browser ping
# is fire-and-forget JS — no Streamlit rerun, so History/Alerts pages don't flicker.
_cfg = config.load()
_wurl = worker_client.base_url(_cfg)
_host = _cfg.get("worker_host", "?")
_online = _worker_online(_wurl)
if _online:
    st.sidebar.markdown(
        f"<div style='padding:8px 10px;border:1px solid #a3e635;border-radius:8px;"
        f"background:rgba(163,230,53,0.06);font-family:JetBrains Mono;font-size:0.74rem;"
        f"margin-bottom:10px'>"
        f"<span style='color:#a3e635'>● WORKER ONLINE</span><br>"
        f"<span style='color:#7d8896'>{_host}</span></div>",
        unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        f"<div style='padding:8px 10px;border:1px solid #26303c;border-radius:8px;"
        f"font-family:JetBrains Mono;font-size:0.74rem;margin-bottom:10px;color:#7d8896'>"
        f"○ WORKER OFFLINE<br><span>{_host} unreachable</span></div>",
        unsafe_allow_html=True)

# Browser-side heartbeat: ping the worker every `dashboard_heartbeat_sec`. `no-cors`
# is fire-and-forget (the worker still receives it; we don't need to read the reply),
# so no CORS setup and no page rerun. While ANY tab is open the worker sees us and
# pauses its own logging.
_hb_ms = int(_cfg.get("dashboard_heartbeat_sec", 8)) * 1000
components.html(
    f"""
    <script>
      const u = "{_wurl}/ping";
      const ping = () => fetch(u, {{mode:'no-cors', cache:'no-store'}}).catch(()=>{{}});
      ping();
      setInterval(ping, {_hb_ms});
    </script>
    """,
    height=0,
)

# ---- render selected page ----
if choice == "Calibration" and role == "admin":
    calibrate_view.render(role=role)
elif choice == "History":
    history_view.render(role=role)
elif choice == "Alerts":
    alerts_view.render(role=role)
else:
    finalise_view.render(role=role)
