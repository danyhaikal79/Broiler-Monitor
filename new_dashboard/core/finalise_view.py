"""
Monitor view — in the split deployment the DASHBOARD does its own inference.

- Upload mode (default): you drop an image/clip; the dashboard runs YOLO + CV here
  on the laptop, logs the reading to the DB, fires alerts, and shows the result.
- Live mode: the dashboard pulls fresh frames from the WORKER on the Jetson
  (core.worker_client, over the LAN by hostname), runs inference here, displays
  them, and logs to the DB throttled to `interval_sec`.

The worker only logs on its OWN when this dashboard is absent. While the dashboard
is open it pings the worker (see app.py) so the worker stands down and the laptop
is the brain.
"""

from __future__ import annotations

import hashlib
import time

import streamlit as st

from . import theme, video, inference, config, cv_feeder, engine, worker_client


def _alert_state():
    """Per-session cooldown timers for the dashboard's alerts (mirrors the worker's)."""
    return st.session_state.setdefault("fin_alert_state", {"last_low": 0.0, "last_heat": 0.0})


def _env_inputs(cfg, prefix="fin_env"):
    """Shared Environment sidebar — IDENTICAL widgets in BOTH Live and Upload modes (same
    keys), so switching modes never leaves a stale / duplicate Environment block. The
    DHT-22 is on the JETSON (port fixed there), so 'Auto' uses the live Jetson reading —
    there is NO COM-port picker on the laptop. Uncheck Auto to type values manually.

    Returns (auto, manual_temp, manual_hum, age). When auto=True the CALLER supplies the
    sensor value (Live: from the frame it already fetched; Upload: via worker_client.fetch_env)."""
    st.sidebar.markdown("### 🌡️ Environment")
    auto = st.sidebar.checkbox("🛰️ Auto — use Jetson sensor", value=True, key=f"{prefix}_auto",
                               help="Use the live DHT-22 reading from the Jetson. "
                                    "Uncheck to enter temperature / humidity manually.")
    mt = mh = None
    if not auto:
        mt = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0, 0.5, key=f"{prefix}_t")
        mh = st.sidebar.number_input("Humidity (% RH)", 0.0, 100.0, 70.0, 1.0, key=f"{prefix}_h")
    age = st.sidebar.number_input("Chicken age (days)", 0, 70, int(config.chicken_age_days(cfg)), 1,
                                  key=f"{prefix}_age")
    return auto, mt, mh, int(age)


def _render_readout(cfg, reading, coverage, images, extras, *, info_caption=""):
    """The full readout: headline cards + Original/Feeder/Feed-mask/Chickens image
    tabs + status panel + daily-coverage bar. `images` is a dict of PIL Images."""
    r = reading or {}
    ex = extras or {}
    imgs = images or {}
    thi = float(r.get("thi_c") or 0.0)
    stress = float(ex.get("thi_stress_factor") if ex.get("thi_stress_factor") is not None else 1.0)
    detected = ex.get("detected")
    fill = float(r.get("fill_ratio") or 0.0)
    count = int(r.get("chicken_count") or 0)
    feed_kg = float(r.get("feed_kg") or 0.0)
    add = float(r.get("feed_to_add_kg") or 0.0)
    required = float(r.get("required_kg") or 0.0)

    if info_caption:
        st.caption(info_caption)

    theme.section("live readout")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.metric_card("Current feed", f"{feed_kg:.2f}", "kg",
                          f"fill {fill*100:.0f}%" if detected else "no feeder", accent=theme.LIME)
    with c2:
        theme.metric_card("Chickens", f"{count}", "",
                          f"conf {float(ex.get('chicken_conf') or 0)*100:.0f}%", accent=theme.CYAN)
    with c3:
        theme.metric_card("THI", f"{thi:.1f}", "°C",
                          "heat stress" if stress < 1 else "comfort zone",
                          accent=theme.AMBER if stress < 1 else theme.LIME)
    with c4:
        if count == 0:
            theme.metric_card("Feed to add", "—", "", "no birds detected", accent=theme.MUTED)
        else:
            theme.metric_card("Feed to add", f"{add:.2f}", "kg",
                              "REFILL NEEDED" if add > 0 else "stocked OK",
                              accent=theme.RED if add > 0 else theme.LIME)

    left, right = st.columns([1.15, 1])
    with left:
        theme.section("vision")
        tabs = st.tabs(["Original", "Feeder", "Feed mask", "Chickens"])
        with tabs[0]:
            st.image(imgs["original"], use_container_width=True) if imgs.get("original") else st.info("No frame.")
        with tabs[1]:
            if imgs.get("feeder"):
                st.image(imgs["feeder"], use_container_width=True, caption=detected or "no feeder detected")
            else:
                st.info("No feeder view.")
        with tabs[2]:
            if imgs.get("feed_mask"):
                st.image(imgs["feed_mask"], use_container_width=True,
                         caption=f"green = feed · fill {fill*100:.1f}%")
            else:
                st.info("No feed mask (feeder not detected).")
        with tabs[3]:
            st.image(imgs["chickens"], use_container_width=True) if imgs.get("chickens") else st.info("No chicken view.")

    with right:
        theme.section("status")
        if not detected:
            st.warning("⚠️ No feeder detected — feed estimate unavailable.")
        e1, e2, e3 = st.columns(3)
        with e1:
            theme.metric_card("Dry-bulb", f"{float(r.get('temperature_c') or 0):.1f}", "°C", accent=theme.TEXT)
        with e2:
            theme.metric_card("Wet-bulb", f"{float(ex.get('wet_bulb_c') or 0):.1f}", "°C", "Stull 2011", accent=theme.TEXT)
        with e3:
            theme.metric_card("Feeder", detected or "—", "",
                              f"conf {float(ex.get('feeder_conf') or 0):.2f}" if detected else "",
                              accent=theme.CYAN)
        if stress < 1.0:
            st.warning(f"🔥 Heat stress · THI {thi:.1f}°C > 26°C → intake at {stress*100:.0f}% of baseline.")
        else:
            st.success("✅ Comfort zone (THI ≤ 26°C) — full Ross 308 intake.")
        st.caption(f"Humidity {float(r.get('humidity_pct') or 0):.0f}% RH")
        if count > 0 and required > 0:
            frac = min(1.0, feed_kg / required)
            theme.section("daily coverage")
            st.progress(frac, text=f"{int(frac*100)}% of today's {required:.2f} kg")


# ----------------------------------------------------------------------------
# Live mode — pull frames from the worker, analyse here, log (throttled)
# ----------------------------------------------------------------------------
def _render_live(cfg, method, method_display, interval):
    theme.section("live monitor · worker camera")
    st.caption(f"Pulling live frames from the worker at **{cfg.get('worker_host')}** and analysing "
               "them here on the laptop. The worker logs on its own only when this dashboard is closed.")

    live_refresh = max(2, int(cfg.get("live_refresh_sec", 5) or 5))
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=live_refresh * 1000, key="fin_live_refresh")
    except ImportError:
        st.caption("⚠️ pip install streamlit-autorefresh for the live view to refresh on its own.")
    if st.button("🔄 Refresh now", key="fin_live_manual"):
        st.rerun()

    auto, mt, mh, age = _env_inputs(cfg)   # same Environment block as Upload mode

    img, meta = worker_client.fetch_frame(cfg)
    if img is None:
        st.warning(f"⏳ {meta}")
        st.info("Make sure **worker.py** is running on the Jetson and you're on the same Wi-Fi. "
                f"You can also open `{worker_client.base_url(cfg)}/health` in a browser to check it.")
        return
    # Auto -> use the sensor reading that rode in with this frame; else the manual values.
    temp, hum = (float(meta["temp"]), float(meta["hum"])) if auto else (float(mt), float(mh))

    cv_bundle = cv_feeder.CVConfigBundle.load(cv_feeder.config_path_for(method))
    try:
        row, coverage, images, extras = engine.analyze_image(
            cfg, cv_bundle, img, method, temp, hum, age)
    except Exception as e:
        st.error(f"⚠️ Inference failed: {e}")
        return

    # Throttle DB logging to interval_sec (we DISPLAY every refresh, but don't spam the DB).
    now = time.time()
    last_log = st.session_state.get("fin_live_last_log", 0.0)
    if now - last_log >= interval:
        try:
            engine.commit_reading(cfg, row, coverage, source="live", alert_state=_alert_state())
            st.session_state["fin_live_last_log"] = now
            log_note = " · ✅ logged"
        except Exception as e:
            log_note = f" · ⚠️ log failed: {e}"
    else:
        log_note = f" · next log in ~{int(interval - (now - last_log))}s"

    src = f"Jetson sensor {temp:.0f}°C / {hum:.0f}% RH" if auto else f"manual {temp:.0f}°C / {hum:.0f}%"
    _render_readout(cfg, row, coverage, images, extras,
                    info_caption=f"🎯 {method_display} · 🛰️ live from worker · {src}{log_note}")


# ----------------------------------------------------------------------------
# Upload mode — analyse + log a file here on the laptop
# ----------------------------------------------------------------------------
def _render_upload(cfg, method, method_display, interval):
    st.sidebar.markdown("### 📤 File")
    uploaded = st.sidebar.file_uploader(
        "Image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv", "webm", "m4v"],
        label_visibility="collapsed", key="fin_upload")
    max_frames = 20
    if uploaded is not None and video.is_video(uploaded.name):
        max_frames = st.sidebar.slider("Frames to analyze", 6, 40, 20, 2, key="fin_maxf")
    auto, mt, mh, age = _env_inputs(cfg)   # same Environment block as Live mode

    st.info("🛈 **Upload mode** — the dashboard analyses your file here, **logs it to the database**, "
            "and shows the result. (The worker on the Jetson stays paused while this dashboard is open.)")

    if uploaded is None:
        st.markdown(
            "<div class='cc-card' style='text-align:center; padding:48px'>"
            "<div style='font-size:42px'>📡</div>"
            "<div class='cc-title' style='margin-top:8px'>Upload to analyze</div>"
            "<div class='sub'>Drop an image or a video clip in the sidebar — it will be "
            "analyzed + logged.</div></div>", unsafe_allow_html=True)
        return

    data = uploaded.getvalue()
    MAX_MB = 200
    if len(data) > MAX_MB * 1024 * 1024:
        st.error(f"File is {len(data) / 1024 / 1024:.0f} MB — please upload under {MAX_MB} MB.")
        return
    kind = "video" if video.is_video(uploaded.name) else "image"

    # Resolve the environment NOW (only when there's a file to analyse — so idle upload
    # mode never touches the Jetson). Auto -> read the Jetson sensor; else manual values.
    if auto:
        e = worker_client.fetch_env(cfg)
        if e:
            temp, hum = float(e["temp"]), float(e["hum"])
        else:
            temp, hum = 25.0, 70.0
            st.warning("⚠️ Couldn't read the Jetson sensor — used 25 °C / 70 %. Uncheck Auto to set manually.")
    else:
        temp, hum = float(mt), float(mh)

    # Re-ANALYSE when the file/method/frames OR the environment changes (so the readout
    # always matches the current temp/hum/age), but LOG to the DB only ONCE per
    # file+method+frames — env tweaks update the display without spamming the database.
    file_sig = (hashlib.md5(data).hexdigest()[:12] + f"_m{method}"
                + (f"_f{max_frames}" if kind == "video" else ""))
    analyze_sig = file_sig + f"_t{float(temp):.1f}_h{float(hum):.0f}_a{int(age)}"
    cache = st.session_state.get("fin_upload_cache")
    if not cache or cache.get("sig") != analyze_sig:
        cv_bundle = cv_feeder.CVConfigBundle.load(cv_feeder.config_path_for(method))
        with st.spinner("Analyzing…"):
            try:
                row, coverage, images, extras = engine.analyze_upload(
                    cfg, cv_bundle, data, kind, method, float(temp), float(hum), int(age), max_frames)
            except Exception as e:
                st.error(f"⚠️ Couldn't analyze this file: {e}")
                return
        if st.session_state.get("fin_upload_logged") != file_sig:
            engine.commit_reading(cfg, row, coverage, source="upload", alert_state=_alert_state())
            st.session_state["fin_upload_logged"] = file_sig
        st.session_state["fin_upload_cache"] = {
            "sig": analyze_sig, "row": row, "coverage": coverage, "images": images, "extras": extras}

    c = st.session_state["fin_upload_cache"]
    theme.section("upload result · analyzed + logged")
    peak = " · 🐔 peak-bird frame" if kind == "video" else ""
    _render_readout(cfg, c["row"], c["coverage"], c["images"], c["extras"],
                    info_caption=f"🎯 Analyzed via **{method_display}**{peak} · ✅ logged to the database")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def render(role: str | None = None, show_header: bool = True):
    if show_header:
        theme.header("BROILER MONITOR", "live feed & flock monitoring", role)

    cfg = config.load()
    interval = int(cfg.get("interval_sec", 600) or 600)

    # ---- Feed-estimation method selector ----
    st.sidebar.markdown("### 🎯 Feed method")
    method_label = st.sidebar.radio(
        "Estimation method", list(inference.METHOD_LABELS.values()),
        label_visibility="collapsed", key="fin_method",
        help="Method 1 = whole-feeder ROI · Method 2 = open-area ROI · Method 3 = demo (dummy farm).")
    method = next((m for m, lbl in inference.METHOD_LABELS.items() if lbl == method_label), 1)
    if not inference.feeder_method_available(method):
        st.sidebar.warning(f"⚠️ {inference.METHOD_LABELS[method]} model not available — using Method 1.")
        method = 1
    method_display = inference.METHOD_LABELS[method]

    # ---- input source ----
    st.sidebar.markdown("### 📥 Input source")
    source = st.sidebar.radio(
        "Input source", ["Upload file", "Live camera"],
        label_visibility="collapsed", key="fin_source")

    if source == "Live camera":
        _render_live(cfg, method, method_display, interval)
    else:
        _render_upload(cfg, method, method_display, interval)
