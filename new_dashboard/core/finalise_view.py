"""
Monitor view — a PURE VIEWER/CONTROLLER. It runs NO inference and never writes to
the database. All YOLO/CV inference + DB writes happen in the worker process.

- Live mode  : sets the worker's camera flag ON and displays what the worker
               publishes (latest numbers + annotated frame), auto-refreshing.
- Upload mode: sets the camera flag OFF (worker camera paused), hands the uploaded
               image/video to the worker via core.control, and shows the worker's
               analyzed + logged result when it comes back (a few seconds later).
"""

from __future__ import annotations

import hashlib
import time

import streamlit as st

from . import theme, video, sensor, inference, config, control


def _render_worker_readout(cfg, reading, coverage, images, extras, *, info_caption=""):
    """The full readout (the old layout): headline cards + Original/Feeder/Feed-mask/
    Chickens image tabs + status panel + daily-coverage bar. Driven entirely by what
    the WORKER published (numbers + the 4 images + extras) — the dashboard infers nothing."""
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


def _render_worker_live(cfg, interval: int):
    """Live mode = display what the worker published (it owns the camera + DB)."""
    theme.section("live monitor · worker")
    st.caption("The **worker** on this device owns the camera and logs to the database. It keeps "
               "running even if you close this page. Switch to **Upload file** to pause its camera.")

    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=4000, key="fin_worker_refresh")
    except ImportError:
        st.caption("⚠️ pip install streamlit-autorefresh for the live view to refresh on its own.")
    if st.button("🔄 Refresh", key="fin_worker_manual"):
        st.rerun()

    latest = control.read_latest()
    if latest is None:
        st.warning("⏳ No worker data yet. Make sure **worker.py** is running on this device — it's "
                   "the process that captures + logs. It should publish a frame within a few seconds "
                   "of entering live mode.")
        return
    if not control.is_fresh(latest["age_s"], interval):
        st.warning(f"⚠️ Worker data is stale ({latest['age_s']:.0f}s old; a new reading is expected "
                   f"every ~{interval}s). The worker may have stopped, or its camera is "
                   f"unavailable (busy/unplugged). Check worker.py on this device.")

    wmethod = int(cfg.get("feeder_method", 1))
    wlabel = inference.METHOD_LABELS.get(wmethod, inference.METHOD_LABELS[1])
    _render_worker_readout(
        cfg, latest.get("reading"), latest.get("coverage"),
        latest.get("images"), latest.get("extras"),
        info_caption=f"🎯 Worker feed method: **{wlabel}** (set in monitor_config.json) · "
                     f"🕒 updated {latest['age_s']:.0f}s ago")


def render(role: str | None = None, show_header: bool = True):
    if show_header:
        theme.header("BROILER MONITOR", "live feed & flock monitoring", role)

    cfg = config.load()
    interval = int(cfg.get("interval_sec", 30) or 30)

    # ---- Feed-estimation method selector (applies to uploads; sent to the worker) ----
    st.sidebar.markdown("### 🎯 Feed method")
    method_label = st.sidebar.radio(
        "Estimation method", list(inference.METHOD_LABELS.values()),
        label_visibility="collapsed", key="fin_method",
        help="Method 1 = whole-feeder ROI · Method 2 = open-area ROI · Method 3 = demo "
             "(dummy farm). Applies to uploads; live mode uses the worker's feeder_method.")
    method = next((m for m, lbl in inference.METHOD_LABELS.items() if lbl == method_label), 1)
    if not inference.feeder_method_available(method):
        st.sidebar.warning(f"⚠️ {inference.METHOD_LABELS[method]} model not available — using Method 1.")
        method = 1
    method_display = inference.METHOD_LABELS[method]

    # ---- input source — the MODE controls the worker's camera. Defaults to the worker's
    # CURRENT state so re-opening the dashboard never accidentally pauses a live worker.
    ctrl = control.read_control()
    st.sidebar.markdown("### 📥 Input source")
    source = st.sidebar.radio(
        "Input source", ["Upload file", "Live camera"],
        index=(1 if ctrl.get("camera_enabled") else 0),
        label_visibility="collapsed", key="fin_source")

    want_live = (source == "Live camera")

    # Live-mode manual environment override — set temp/humidity/age yourself instead of
    # the worker's DHT-22 / fallback (handy for demos with no sensor). Defaults seed from
    # the current on-disk override so re-opening the dashboard doesn't reset it.
    cur_ovr = ctrl.get("env_override")
    env_override = None
    if want_live:
        st.sidebar.markdown("### 🌡️ Environment")
        if st.sidebar.checkbox("Override sensor (set manually)", value=(cur_ovr is not None),
                               key="fin_live_override",
                               help="Use your own temp/humidity/age instead of the worker's "
                                    "DHT-22 / fallback. Useful with no sensor attached."):
            ot = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0,
                                         float(cur_ovr.get("temp", 25.0)) if cur_ovr else 25.0, 0.5,
                                         key="fin_ovr_t")
            oh = st.sidebar.number_input("Humidity (% RH)", 0.0, 100.0,
                                         float(cur_ovr.get("hum", 70.0)) if cur_ovr else 70.0, 1.0,
                                         key="fin_ovr_h")
            oa = st.sidebar.number_input("Chicken age (days)", 0, 60,
                                         int(cur_ovr.get("age")) if cur_ovr else int(config.chicken_age_days(cfg)),
                                         1, key="fin_ovr_a")
            env_override = {"temp": float(ot), "hum": float(oh), "age": int(oa)}

    # Tell the worker the desired state (camera flag + env override). Write only when it
    # differs from what's on disk — so no churn, and it also reconciles another tab/process.
    desired = {"camera_enabled": want_live, "env_override": env_override}
    current = {"camera_enabled": bool(ctrl.get("camera_enabled")), "env_override": cur_ovr}
    if desired != current:
        control.set_control(want_live, env_override)

    # ===================== LIVE: view the worker (it owns the camera + DB) =====================
    if source == "Live camera":
        _render_worker_live(cfg, interval)
        return

    # ===================== UPLOAD: hand the file to the worker; show its result =================
    # File uploader sits RIGHT UNDER the Input-source options; Environment goes below it.
    st.sidebar.markdown("### 📤 File")
    uploaded = st.sidebar.file_uploader(
        "Image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv", "webm", "m4v"],
        label_visibility="collapsed", key="fin_upload")
    max_frames = 20
    if uploaded is not None and video.is_video(uploaded.name):
        max_frames = st.sidebar.slider("Frames to analyze", 6, 40, 20, 2, key="fin_maxf")
    temp, hum, age = sensor.sidebar_env_inputs("fin")   # Environment section, below the uploader

    st.info("🛈 **Upload mode** — the worker's camera is **paused**. The worker analyzes your file, "
            "**logs it to the database**, and returns the result here (takes a few seconds). The "
            "dashboard itself runs no inference.")

    if uploaded is None:
        st.markdown(
            "<div class='cc-card' style='text-align:center; padding:48px'>"
            "<div style='font-size:42px'>📡</div>"
            "<div class='cc-title' style='margin-top:8px'>Upload to analyze</div>"
            "<div class='sub'>Drop an image or a video clip in the sidebar — the worker will "
            "analyze + log it.</div></div>", unsafe_allow_html=True)
        return

    data = uploaded.getvalue()
    MAX_MB = 64
    if len(data) > MAX_MB * 1024 * 1024:
        st.error(f"File is {len(data) / 1024 / 1024:.0f} MB — please upload under {MAX_MB} MB.")
        return
    kind = "video" if video.is_video(uploaded.name) else "image"

    # Request id = file content + method (+frame count) ONLY — NOT the live sensor env,
    # so DHT-22 Live-mode drift can't churn the id and re-submit/re-log the same file.
    req_id = (hashlib.md5(data).hexdigest()[:12] + f"_m{method}"
              + (f"_f{max_frames}" if kind == "video" else ""))
    if st.session_state.get("fin_upload_submitted") != req_id:
        try:
            control.submit_upload(req_id, kind, data, method=method, temp=float(temp),
                                  hum=float(hum), age=int(age), max_frames=max_frames)
            st.session_state["fin_upload_submitted"] = req_id   # mark only AFTER a successful write
            st.session_state["fin_upload_wait_t0"] = None
        except Exception as e:
            st.error(f"⚠️ Couldn't hand the file to the worker (will retry): {e}")
            return

    res = control.read_upload_result()
    if res is None or res.get("id") != req_id:
        # waiting — poll, but time out so a missing/stopped worker doesn't hang the UI forever
        t0 = st.session_state.get("fin_upload_wait_t0") or time.time()
        st.session_state["fin_upload_wait_t0"] = t0
        if time.time() - t0 > 30:
            st.error("⚠️ No result after 30s — is **worker.py** running on this device? "
                     "Re-select the file, or check the worker.")
            return
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=2000, key="fin_upload_refresh")
        except ImportError:
            if st.button("🔄 Check for result", key="fin_upload_check"):
                st.rerun()
        st.info("⏳ Sent to the worker — analyzing + logging… (make sure **worker.py** is running).")
        return

    if res.get("error"):
        st.error(f"⚠️ The worker couldn't process this file: {res['error']}")
        return

    theme.section("upload result · analyzed + logged by the worker")
    peak_note = " · 🐔 peak-bird frame" if kind == "video" else ""
    _render_worker_readout(
        cfg, res.get("reading"), res.get("coverage"), res.get("images"), res.get("extras"),
        info_caption=f"🎯 Analyzed via **{method_display}**{peak_note} · "
                     f"✅ logged to the database by the worker")
