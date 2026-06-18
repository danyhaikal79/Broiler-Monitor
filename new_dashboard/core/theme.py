"""
Dark command-center theme + UI helpers.

A single inject_css() call restyles Streamlit; helper functions render custom
HTML components (header, metric cards, status pills, section titles) so the
dashboard looks like a purpose-built IoT console rather than default Streamlit.
"""

from __future__ import annotations

import streamlit as st

# Palette
BG = "#0b0f14"          # near-black slate
PANEL = "#141a22"       # card background
PANEL_2 = "#1b232d"     # raised card
BORDER = "#26303c"
TEXT = "#e6edf3"
MUTED = "#7d8896"
LIME = "#a3e635"        # primary accent
LIME_DIM = "#5f7a1f"
AMBER = "#fbbf24"       # secondary accent
CYAN = "#38bdf8"
RED = "#f87171"


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

    /* ---- base ---- */
    .stApp {{
        background:
            radial-gradient(1200px 600px at 85% -10%, rgba(163,230,53,0.06), transparent 60%),
            radial-gradient(900px 500px at -10% 110%, rgba(56,189,248,0.05), transparent 60%),
            {BG};
        color: {TEXT};
        font-family: 'Space Grotesk', system-ui, sans-serif;
    }}
    /* hide default chrome */
    #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.2rem; max-width: 1300px; }}

    /* ---- kill the rerun "dim" flash (live mode refreshes every 5s) ---- */
    /* Streamlit fades stale elements to ~0.4 opacity during a rerun -> override it */
    [data-stale="true"], .stApp [data-stale="true"], div[data-stale] {{
        opacity: 1 !important; transition: none !important;
    }}
    /* hide the top-right "Running..." status widget */
    [data-testid="stStatusWidget"] {{ display: none !important; }}
    /* hide the thin top loading/progress bar shown on rerun */
    div[data-testid="stDecoration"] {{ display: none !important; }}
    .stApp [data-testid="stAppViewBlockContainer"] {{ transition: none !important; }}

    h1,h2,h3,h4 {{ font-family: 'Space Grotesk', sans-serif; color: {TEXT}; letter-spacing: -0.01em; }}

    /* sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PANEL} 0%, {BG} 100%);
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

    /* inputs */
    .stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
        background: {PANEL_2} !important; color: {TEXT} !important;
        border: 1px solid {BORDER} !important; border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: {LIME} !important; box-shadow: 0 0 0 2px rgba(163,230,53,0.15) !important;
    }}

    /* buttons */
    .stButton > button, [data-testid="stFormSubmitButton"] button {{
        background: linear-gradient(180deg, {LIME} 0%, #8fd131 100%);
        color: #0a0f06; font-weight: 700; border: none; border-radius: 10px;
        padding: 0.5rem 1.1rem; letter-spacing: 0.02em;
        box-shadow: 0 4px 18px rgba(163,230,53,0.25); transition: transform .08s ease, box-shadow .2s ease;
    }}
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
        transform: translateY(-1px); box-shadow: 0 6px 24px rgba(163,230,53,0.4);
    }}

    /* bordered container -> glassy card (used for the login form) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(160deg, {PANEL} 0%, {PANEL_2} 100%);
        border: 1px solid {BORDER} !important; border-radius: 18px;
        box-shadow: 0 30px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03);
    }}
    /* tighten form spacing */
    [data-testid="stForm"] {{ border: none !important; padding: 0 !important; }}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; color: {MUTED}; border-radius: 8px 8px 0 0;
        font-family: 'Space Grotesk'; font-weight: 600; padding: 8px 14px;
    }}
    .stTabs [aria-selected="true"] {{ color: {LIME}; border-bottom: 2px solid {LIME}; }}

    /* progress */
    .stProgress > div > div > div {{ background: linear-gradient(90deg,{LIME},{AMBER}); }}

    /* alerts -> dark glassy */
    div[data-testid="stAlert"] {{
        background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 12px; color: {TEXT};
    }}

    /* expander */
    details {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; }}
    summary {{ color: {TEXT}; font-weight: 600; }}

    /* custom components */
    .cc-header {{
        display:flex; align-items:center; justify-content:space-between;
        padding: 16px 22px; margin-bottom: 18px; border-radius: 16px;
        background: linear-gradient(120deg, {PANEL} 0%, {PANEL_2} 100%);
        border: 1px solid {BORDER};
        box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
    }}
    .cc-brand {{ display:flex; align-items:center; gap:12px; }}
    .cc-logo {{
        width: 38px; height: 38px; border-radius: 10px;
        background: radial-gradient(circle at 30% 30%, {LIME}, {LIME_DIM});
        box-shadow: 0 0 18px rgba(163,230,53,0.5);
        display:flex; align-items:center; justify-content:center; font-size:20px;
    }}
    .cc-title {{ font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }}
    .cc-sub {{ font-size: 0.72rem; color: {MUTED}; font-family:'JetBrains Mono'; letter-spacing:0.08em; text-transform:uppercase; }}
    .cc-badge {{
        font-family:'JetBrains Mono'; font-size:0.72rem; font-weight:700; letter-spacing:0.1em;
        padding: 6px 12px; border-radius: 999px; text-transform:uppercase;
        border:1px solid {BORDER}; background:{PANEL_2};
    }}
    .cc-badge.admin {{ color:{AMBER}; border-color: rgba(251,191,36,0.4); box-shadow:0 0 14px rgba(251,191,36,0.18); }}
    .cc-badge.staff {{ color:{CYAN}; border-color: rgba(56,189,248,0.4); box-shadow:0 0 14px rgba(56,189,248,0.18); }}
    .cc-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:{LIME};
               box-shadow:0 0 8px {LIME}; margin-right:7px; animation: ccpulse 1.8s infinite; }}
    @keyframes ccpulse {{ 0%{{opacity:1}} 50%{{opacity:0.3}} 100%{{opacity:1}} }}

    .cc-card {{
        background: linear-gradient(160deg, {PANEL} 0%, {PANEL_2} 100%);
        border: 1px solid {BORDER}; border-radius: 16px; padding: 18px 20px;
        position: relative; overflow: hidden; height: 100%;
        transition: transform .12s ease, border-color .2s ease;
    }}
    .cc-card:hover {{ transform: translateY(-2px); border-color: rgba(163,230,53,0.35); }}
    .cc-card .lab {{ font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; color:{MUTED}; font-family:'JetBrains Mono'; }}
    .cc-card .val {{ font-family:'JetBrains Mono'; font-weight:700; font-size:2.0rem; line-height:1.1; margin-top:6px; }}
    .cc-card .unit {{ font-size:0.95rem; color:{MUTED}; font-weight:600; }}
    .cc-card .sub {{ font-size:0.72rem; color:{MUTED}; margin-top:6px; }}
    .cc-card .spark {{ position:absolute; right:-10px; top:-10px; width:70px; height:70px; border-radius:50%;
                       background: radial-gradient(circle, rgba(163,230,53,0.18), transparent 70%); }}

    .cc-section {{ font-family:'JetBrains Mono'; font-size:0.74rem; letter-spacing:0.16em; text-transform:uppercase;
                   color:{MUTED}; margin: 22px 0 10px; display:flex; align-items:center; gap:10px; }}
    .cc-section::after {{ content:''; flex:1; height:1px; background:{BORDER}; }}

    .cc-login-card {{
        max-width: 420px; margin: 6vh auto 0; padding: 34px 32px; border-radius: 20px;
        background: linear-gradient(160deg, {PANEL} 0%, {PANEL_2} 100%);
        border:1px solid {BORDER}; box-shadow: 0 30px 80px rgba(0,0,0,0.5);
    }}
    </style>
    """, unsafe_allow_html=True)


def header(title: str, subtitle: str, role: str | None = None, centered: bool = False):
    badge = ""
    if role and not centered:
        badge = f'<span class="cc-badge {role}">{role}</span>'
    justify = "center" if centered else "space-between"
    text_align = "text-align:center;" if centered else ""
    st.markdown(f"""
    <div class="cc-header" style="justify-content:{justify}">
      <div class="cc-brand" style="{text_align}">
        <div class="cc-logo">🐔</div>
        <div>
          <div class="cc-title">{title}</div>
          <div class="cc-sub"><span class="cc-dot"></span>{subtitle}</div>
        </div>
      </div>
      {badge}
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, unit: str = "", sub: str = "", accent: str = LIME):
    st.markdown(f"""
    <div class="cc-card">
      <div class="spark"></div>
      <div class="lab">{label}</div>
      <div class="val" style="color:{accent}">{value}<span class="unit"> {unit}</span></div>
      <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section(label: str):
    st.markdown(f'<div class="cc-section">{label}</div>', unsafe_allow_html=True)
