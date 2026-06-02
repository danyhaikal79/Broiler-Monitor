"""
Simple session-based auth for the super dashboard (FYP demo).

Hardcoded accounts. NOT secure — passwords are plaintext in code. Fine for a
local FYP demo; do not use as-is in production.
"""

from __future__ import annotations

import streamlit as st

from . import theme

# username -> {password, role, display}
ACCOUNTS = {
    "admin": {"password": "admin123", "role": "admin", "display": "Administrator"},
    "staff": {"password": "staff123", "role": "staff", "display": "Farm Staff"},
}


def is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user"))


def current_user() -> dict | None:
    u = st.session_state.get("auth_user")
    return ACCOUNTS.get(u) | {"username": u} if u else None


def current_role() -> str | None:
    u = current_user()
    return u["role"] if u else None


def logout():
    st.session_state.pop("auth_user", None)
    st.session_state.pop("nav_page", None)
    st.rerun()


def login_screen():
    """Render a centered login card. Sets session on success."""
    # Header + card share the same narrow centered column so they line up and
    # the header isn't a full-width bar.
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        theme.header("BROILER MONITOR", "secure access · authenticate", centered=True)
        with st.container(border=True):
            st.markdown(
                "<div style='display:flex;align-items:center;gap:12px;margin-bottom:2px'>"
                "<div style='font-size:30px'>🔐</div>"
                "<div><div style='font-size:1.35rem;font-weight:700'>Sign in</div>"
                "<div style='font-size:0.8rem;color:#7d8896'>Authenticate to access the monitor</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-size:0.82rem;color:#7d8896;margin:6px 0 14px'>"
                "🛡️ Admins get <b style='color:#fbbf24'>calibration + monitoring</b> · "
                "staff get <b style='color:#38bdf8'>monitoring</b>.</div>",
                unsafe_allow_html=True,
            )

            with st.form("login_form", clear_on_submit=False, border=False):
                user = st.text_input("Username", key="login_user", placeholder="admin / staff")
                pwd = st.text_input("Password", type="password", key="login_pwd",
                                    placeholder="••••••••")
                go = st.form_submit_button("Sign in  →", use_container_width=True)

            if go:
                acc = ACCOUNTS.get(user.strip())
                if acc and pwd == acc["password"]:
                    st.session_state["auth_user"] = user.strip()
                    st.session_state["nav_page"] = "Monitor"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.caption("demo · admin / admin123  ·  staff / staff123")
