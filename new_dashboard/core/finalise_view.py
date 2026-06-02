"""
Finalise (monitoring) view — handles IMAGE or VIDEO upload, dark command-center styled.
Video: sample frames -> analyze each -> trend charts + scrub to inspect any frame.
"""

from __future__ import annotations

import hashlib
import io

import streamlit as st
from PIL import Image

from . import theme, pipeline, cv_feeder, logic, video, sensor


# ----------------------------------------------------------------------------
# Shared single-frame readout (used by image mode AND the scrubbed video frame)
# ----------------------------------------------------------------------------
def _render_readout(img: Image.Image, b):
    pred, feeder, chicken, cv_meas = b.pred, b.feeder, b.chicken, b.cv_meas

    theme.section("live readout")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.metric_card("Current feed", f"{pred.current_food_kg:.2f}", "kg",
                          f"fill {cv_meas.fill_ratio*100:.0f}%" if cv_meas else "no feeder",
                          accent=theme.LIME)
    with c2:
        theme.metric_card("Chickens", f"{pred.chicken_count}", "",
                          f"conf {chicken.avg_confidence*100:.0f}%", accent=theme.CYAN)
    with c3:
        theme.metric_card("THI", f"{pred.thi_c:.1f}", "°C",
                          "heat stress" if pred.thi_stress_factor < 1 else "comfort zone",
                          accent=theme.AMBER if pred.thi_stress_factor < 1 else theme.LIME)
    with c4:
        if pred.chicken_count == 0:
            theme.metric_card("Feed to add", "—", "", "no birds detected", accent=theme.MUTED)
        else:
            need = pred.feed_to_add_kg > 0
            theme.metric_card("Feed to add", f"{pred.feed_to_add_kg:.2f}", "kg",
                              "REFILL NEEDED" if need else "stocked OK",
                              accent=theme.RED if need else theme.LIME)

    left, right = st.columns([1.15, 1])
    with left:
        theme.section("vision")
        tabs = st.tabs(["Original", "Feeder", "Feed mask", "Chickens"])
        with tabs[0]:
            st.image(img, use_container_width=True)
        with tabs[1]:
            st.image(feeder.annotated_image or img, use_container_width=True,
                     caption=f"{b.detected_feeder_type or 'no feeder detected'}")
        with tabs[2]:
            if cv_meas and cv_meas.mask_image is not None:
                st.image(cv_meas.mask_image, use_container_width=True,
                         caption=f"green = feed · fill {cv_meas.fill_ratio*100:.1f}%")
            else:
                st.info("No feed mask (feeder not detected).")
        with tabs[3]:
            st.image(chicken.annotated_image or img, use_container_width=True)

    with right:
        theme.section("status")
        if not b.detected_feeder_type:
            st.warning("⚠️ No feeder detected — feed estimate unavailable.")
        e1, e2, e3 = st.columns(3)
        with e1: theme.metric_card("Dry-bulb", f"{pred.temperature_c:.1f}", "°C", accent=theme.TEXT)
        with e2: theme.metric_card("Wet-bulb", f"{pred.wet_bulb_c:.1f}", "°C", "Stull 2011", accent=theme.TEXT)
        with e3: theme.metric_card("Feeder", b.detected_feeder_type or "—", "",
                                   f"conf {feeder.feeder_type_conf:.2f}" if b.detected_feeder_type else "",
                                   accent=theme.CYAN)
        if pred.thi_stress_factor < 1.0:
            st.warning(f"🔥 Heat stress · THI {pred.thi_c:.1f}°C > 21°C → intake at "
                       f"{pred.thi_stress_factor*100:.0f}% of baseline.")
        else:
            st.success("✅ Thermoneutral zone — full Ross 308 intake.")
        if pred.chicken_count > 0 and pred.flock_daily_required_kg > 0:
            cov = min(1.0, pred.current_food_kg / pred.flock_daily_required_kg)
            theme.section("daily coverage")
            st.progress(cov, text=f"{int(cov*100)}% of today's {pred.flock_daily_required_kg:.2f} kg")


def _video_signature(video_bytes, temp, hum, age, max_frames) -> str:
    h = hashlib.md5(video_bytes).hexdigest()[:16]
    return f"{h}_{temp}_{hum}_{age}_{max_frames}"


def render(role: str | None = None, show_header: bool = True):
    if show_header:
        theme.header("BROILER MONITOR", "live feed & flock monitoring", role)

    cv_bundle = cv_feeder.CVConfigBundle.load()

    st.sidebar.markdown("### 📥 Input source")
    source = st.sidebar.radio("Input source", ["Upload file", "Live camera"],
                              label_visibility="collapsed", key="fin_source")

    # Input-source-specific controls render RIGHT HERE (under the picker), so the
    # camera button isn't buried below the Environment section.
    uploaded = None
    cam_idx, auto, secs, cap_now = 0, False, 5, False
    if source == "Upload file":
        uploaded = st.sidebar.file_uploader(
            "Image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv", "webm", "m4v"],
            label_visibility="collapsed", key="fin_upload",
        )
    else:  # Live camera
        st.sidebar.markdown("### 🎥 Camera")
        cam_idx = st.sidebar.number_input("Camera index", 0, 8, 0, 1, key="fin_cam_idx",
                                          help="0 = default camera. Try 1/2 if you have multiple.")
        cap_now = st.sidebar.button("📸 Capture now", use_container_width=True)
        auto = st.sidebar.checkbox("Auto-capture", value=False, key="fin_auto",
                                   help="Grab a frame automatically on an interval.")
        if auto:
            secs = st.sidebar.slider("Interval (sec)", 2, 30, 5, 1, key="fin_auto_secs")

    temp, hum, age = sensor.sidebar_env_inputs("fin")

    pipeline.warm_models()

    # ========================= LIVE CAMERA =========================
    if source == "Live camera":
        def _grab_and_store():
            frame = video.capture_frame(int(cam_idx))
            if frame is None:
                st.session_state["fin_cam_err"] = True
            else:
                st.session_state["fin_cam_err"] = False
                st.session_state["fin_cam_frame"] = frame

        if cap_now:
            _grab_and_store()

        # Auto-capture: re-run this fragment on an interval (Streamlit >=1.37)
        if auto:
            @st.fragment(run_every=f"{secs}s")
            def _auto_frag():
                _grab_and_store()
                st.caption(f"🔴 LIVE · auto-capturing every {secs}s from camera {cam_idx}")
            _auto_frag()

        if st.session_state.get("fin_cam_err"):
            st.error(f"Could not read camera index {cam_idx}. Check it's connected and not in use "
                     "by another app. On the Jetson, the app must run ON the Jetson.")
            return
        if "fin_cam_frame" not in st.session_state:
            st.markdown(
                "<div class='cc-card' style='text-align:center; padding:48px'>"
                "<div style='font-size:42px'>🎥</div>"
                "<div class='cc-title' style='margin-top:8px'>Live camera ready</div>"
                "<div class='sub'>Press <b>📸 Capture now</b> (or enable auto-capture) to grab a frame "
                "from the camera attached to this machine.</div></div>", unsafe_allow_html=True)
            return

        img = st.session_state["fin_cam_frame"]
        with st.spinner("Analyzing frame…"):
            b = pipeline.run_all(img, cv_bundle, temperature_c=temp, humidity_pct=hum, age_days=age)
        _render_readout(img, b)
        return

    # ========================= UPLOAD (image/video) =========================
    if uploaded is None:
        st.markdown(
            "<div class='cc-card' style='text-align:center; padding:48px'>"
            "<div style='font-size:42px'>📡</div>"
            "<div class='cc-title' style='margin-top:8px'>Awaiting camera input</div>"
            "<div class='sub'>Upload an image or a video clip from the sidebar to start.</div>"
            "</div>", unsafe_allow_html=True)
        return

    # ========================= VIDEO =========================
    if video.is_video(uploaded.name):
        st.sidebar.markdown("### 🎞️ Video")
        max_frames = st.sidebar.slider("Frames to analyze", 6, 40, 20, 2, key="fin_maxf")
        vbytes = uploaded.getvalue()

        sig = _video_signature(vbytes, temp, hum, age, max_frames)
        if st.session_state.get("fin_vid_sig") != sig:
            frames = video.sample_frames(vbytes, max_frames=max_frames)
            if not frames:
                st.error("Could not read frames from this video.")
                return
            results = []
            prog = st.progress(0.0, text="Analyzing frames…")
            for i, (ts, fr) in enumerate(frames):
                b = pipeline.run_all(fr, cv_bundle, temperature_c=temp, humidity_pct=hum, age_days=age)
                results.append((ts, fr, b))
                prog.progress((i + 1) / len(frames), text=f"Analyzing frame {i+1}/{len(frames)}")
            prog.empty()
            st.session_state["fin_vid_sig"] = sig
            st.session_state["fin_vid_results"] = results
        results = st.session_state["fin_vid_results"]

        # Timeline trend
        theme.section("timeline")
        trend = [{
            "t_sec": round(ts, 1),
            "feed_kg": round(b.pred.current_food_kg, 3),
            "chickens": b.pred.chicken_count,
        } for ts, _f, b in results]
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown("**Feed level (kg) over time**")
            st.line_chart(trend, x="t_sec", y="feed_kg", height=200, color=theme.LIME)
        with tcol2:
            st.markdown("**Chicken count over time**")
            st.line_chart(trend, x="t_sec", y="chickens", height=200, color=theme.CYAN)

        # Aggregate summary cards
        feeds = [b.pred.current_food_kg for _t, _f, b in results]
        counts = [b.pred.chicken_count for _t, _f, b in results]
        theme.section("clip summary")
        s1, s2, s3, s4 = st.columns(4)
        with s1: theme.metric_card("Avg feed", f"{sum(feeds)/len(feeds):.2f}", "kg", accent=theme.LIME)
        with s2: theme.metric_card("Min feed", f"{min(feeds):.2f}", "kg", "lowest in clip", accent=theme.AMBER)
        with s3: theme.metric_card("Peak birds", f"{max(counts)}", "", "max in clip", accent=theme.CYAN)
        with s4: theme.metric_card("Frames", f"{len(results)}", "", "analyzed", accent=theme.TEXT)

        # Default to the PEAK-chicken frame: across the clip, the frame with the most
        # detections is the best flock-size estimate (other frames undercount due to
        # occlusion / birds moving out of view).
        peak_idx = max(range(len(results)), key=lambda i: results[i][2].pred.chicken_count)

        theme.section("inspect frame")
        st.caption(f"🐔 Headline uses the **peak-count frame** "
                   f"(frame {peak_idx+1}, {results[peak_idx][2].pred.chicken_count} chickens). "
                   "Scrub to inspect any other frame.")
        idx = st.slider("Frame", 0, len(results) - 1, peak_idx, key="fin_scrub",
                        format="frame %d")
        ts, fr, b = results[idx]
        peak_tag = " · 🐔 peak-count frame" if idx == peak_idx else ""
        st.caption(f"⏱️ t = {ts:.1f}s  ·  frame {idx+1}/{len(results)}{peak_tag}")
        _render_readout(fr, b)
        return

    # ========================= IMAGE =========================
    img = Image.open(io.BytesIO(uploaded.getvalue()))
    with st.spinner("Analyzing frame…"):
        b = pipeline.run_all(img, cv_bundle, temperature_c=temp, humidity_pct=hum, age_days=age)
    _render_readout(img, b)

    with st.expander("🔍 Computation breakdown"):
        pred, feeder, cv_meas = b.pred, b.feeder, b.cv_meas
        st.json({
            "environment": {"wet_bulb_C": round(pred.wet_bulb_c, 2), "THI_C": round(pred.thi_c, 2),
                            "stress_factor": round(pred.thi_stress_factor, 3)},
            "vision": {"feeder_type": b.detected_feeder_type,
                       "feeder_conf": round(feeder.feeder_type_conf, 3),
                       "cv_fill_ratio": round(cv_meas.fill_ratio, 4) if cv_meas else None,
                       "chicken_count": pred.chicken_count},
            "feed_math": {"baseline_g_per_bird": logic.ROSS308_DAILY_INTAKE_G.get(min(pred.age_days, 56), 0),
                          "stress_adjusted_g_per_bird": round(pred.per_bird_daily_g, 1),
                          "flock_required_kg": round(pred.flock_daily_required_kg, 3),
                          "current_food_kg": round(pred.current_food_kg, 3),
                          "feed_to_add_kg": round(pred.feed_to_add_kg, 3)},
        })
