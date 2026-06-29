"""
Calibrate (admin) view — feed-detection tuning + calibration tools, themed.

Lets an admin: pick a feeder type, tune its combined HSV+texture detection,
batch-preview masks across known-weight photos, and add/manage calibration
samples. Uses the YOLO feeder polygon as the ROI (cached per image).
"""

from __future__ import annotations

import io

import numpy as np
import streamlit as st
from PIL import Image

from . import theme, inference, cv_feeder, logic, video, sensor

FEEDER_TYPES = cv_feeder.FEEDER_TYPES


@st.cache_data(show_spinner=False)
def _yolo_for(img_bytes: bytes, method: int = 1):
    """Cache YOLO feeder result per (image, method) so slider moves don't re-run YOLO."""
    inference.load_feeder_model(method)
    pil = Image.open(io.BytesIO(img_bytes))
    fr = inference.run_feeder(pil, conf=0.1, method=method)
    return {
        "polygon": fr.feeder_polygon.tolist() if fr.feeder_polygon is not None else None,
        "bbox": list(fr.feeder_bbox) if fr.feeder_bbox else None,
        "type": fr.feeder_type,
        "conf": float(fr.feeder_type_conf),
    }


@st.cache_data(show_spinner=False)
def _frames_cached(video_bytes: bytes, max_frames: int):
    """Sample + JPEG-encode frames so they cache cheaply across reruns."""
    frames = video.sample_frames(video_bytes, max_frames=max_frames)
    out = []
    for ts, im in frames:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        out.append((round(ts, 2), buf.getvalue()))
    return out


def _measure(img, cv_bundle, feeder_type, yres):
    poly = np.array(yres["polygon"], dtype=float) if yres and yres["polygon"] else None
    bbox = tuple(yres["bbox"]) if yres and yres["bbox"] else None
    return cv_feeder.measure(img, cv_bundle, feeder_type,
                             yolo_polygon=poly, yolo_bbox=bbox, mask_dilation_px=0)


def render(role: str | None = None, show_header: bool = True):
    if show_header:
        theme.header("CALIBRATION CONSOLE", "feed-detection tuning · admin", role)

    # ---- which method's config are we calibrating? ----
    st.sidebar.markdown("### 🎯 Method")
    m_label = st.sidebar.radio("Method", list(inference.METHOD_LABELS.values()),
                               label_visibility="collapsed", key="cal_method",
                               help="Each method has its own CV config (cv_config*.json). "
                                    "Calibrate them separately. Method 3 = demo / dummy farm.")
    method = next((m for m, lbl in inference.METHOD_LABELS.items() if lbl == m_label), 1)
    if not inference.feeder_method_available(method):
        st.sidebar.warning(f"⚠️ {inference.METHOD_LABELS[method]} model not available — you can still "
                           "pre-tune its config, but YOLO ROI previews need a trained model.")
    cfg_path = cv_feeder.config_path_for(method)
    cv_bundle = cv_feeder.CVConfigBundle.load(cfg_path, method)

    # ---- sidebar: which feeder + detection config ----
    st.sidebar.markdown("### ⚙️ Config target")
    active = st.sidebar.selectbox("Feeder type to edit",
                                  list(cv_feeder.feeder_types_for(method)), key="cal_active")
    cfg = cv_bundle.get(active)
    mk = f"{active}_m{method}"   # widget-key suffix: per feeder AND per method (no cross-bleed)
    previews_ok = inference.feeder_method_available(method)   # YOLO ROI previews need this method's model

    st.sidebar.caption(f"ROI = YOLO polygon (auto). Current saved ROI box: "
                       f"({cfg.roi_x1_pct:.0f},{cfg.roi_y1_pct:.0f})→({cfg.roi_x2_pct:.0f},{cfg.roi_y2_pct:.0f})")

    methods = ["hsv", "texture", "combined"]
    labels = {"hsv": "HSV color", "texture": "Texture", "combined": "Combined (HSV + texture)"}
    cfg.detect_method = st.sidebar.radio(
        "Detection method", methods,
        index=methods.index(cfg.detect_method) if cfg.detect_method in methods else 2,
        format_func=lambda m: labels[m], key=f"cal_m_{mk}",
    )

    if cfg.detect_method in ("hsv", "combined"):
        st.sidebar.markdown("**HSV thresholds**")
        cfg.hsv_h_min = st.sidebar.slider("Hue min", 0, 179, cfg.hsv_h_min, key=f"hmn_{mk}")
        cfg.hsv_h_max = st.sidebar.slider("Hue max", 0, 179, cfg.hsv_h_max, key=f"hmx_{mk}")
        cfg.hsv_s_min = st.sidebar.slider("Sat min", 0, 255, cfg.hsv_s_min, key=f"smn_{mk}")
        cfg.hsv_s_max = st.sidebar.slider("Sat max", 0, 255, cfg.hsv_s_max, key=f"smx_{mk}")
        cfg.hsv_v_min = st.sidebar.slider("Val min", 0, 255, cfg.hsv_v_min, key=f"vmn_{mk}")
        cfg.hsv_v_max = st.sidebar.slider("Val max", 0, 255, cfg.hsv_v_max, key=f"vmx_{mk}")
    if cfg.detect_method in ("texture", "combined"):
        st.sidebar.markdown("**Texture params**")
        cfg.texture_threshold = st.sidebar.slider("Texture sensitivity", 2.0, 40.0,
                                                   float(cfg.texture_threshold), 1.0, key=f"tt_{mk}")
        cfg.texture_window = st.sidebar.slider("Texture window", 3, 15,
                                               int(cfg.texture_window), 2, key=f"tw_{mk}")

    if st.sidebar.button("💾 Save config", use_container_width=True):
        cv_bundle.set(active, cfg)
        cv_bundle.save(cfg_path)
        st.sidebar.success(f"Saved {active} config")

    temp, hum, age = sensor.sidebar_env_inputs("cal")

    theme.section(f"editing · {active}")

    # ---- live calibration summary card (kg prediction = piecewise interpolation) ----
    mono = cv_feeder.calibration_monotonic(cfg.samples)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        theme.metric_card("Method", labels[cfg.detect_method], "", accent=theme.AMBER)
    with cc2:
        theme.metric_card("Samples", f"{len(cfg.samples)}", "", "connect-the-dots points", accent=theme.CYAN)
    with cc3:
        theme.metric_card("Calibration", "✅ OK" if mono else "⚠️ check",
                          "", "monotonic" if mono else "non-monotonic samples",
                          accent=theme.LIME if mono else theme.RED)

    # ============================================================
    #  Batch HSV/texture tuning + calibration
    # ============================================================
    theme.section("batch tuning & calibration")
    st.caption("Upload several known-weight photos of THIS feeder. Masks update live as you "
               "move the sliders. Tune until feed is captured cleanly, then add all as samples.")
    batch = st.file_uploader("Calibration photos (multiple)", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key=f"cal_batch_{mk}")

    if batch and not previews_ok:
        st.info("ℹ️ This method's model isn't available yet — mask previews are disabled (they need "
                "the YOLO ROI). You can still edit + Save threshold values above; add calibration "
                "samples after the model is trained.")
    elif batch:
        per_row = st.slider("Photos per row", 1, 5, 3, key="cal_perrow")
        processed = []
        with st.spinner(f"Processing {len(batch)} photos…"):
            for f in batch:
                data = f.getvalue()
                meas = _measure(Image.open(io.BytesIO(data)), cv_bundle, active, _yolo_for(data, method))
                processed.append((f.name, meas))

        for start in range(0, len(processed), per_row):
            cols = st.columns(per_row)
            for ci, (name, meas) in enumerate(processed[start:start + per_row]):
                with cols[ci]:
                    st.image(meas.mask_image, use_container_width=True,
                             caption=f"{name[:18]} · fill {meas.fill_ratio*100:.1f}%")
                    st.number_input("known kg", 0.0, 10.0, 0.0, 0.1,
                                    key=f"cal_kg_{mk}_{start+ci}")

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button(f"➕ Add all to {active}", use_container_width=True):
                cap = logic.FEEDER_MAX_KG.get(active, 10.0)
                added = 0
                for i, (_, meas) in enumerate(processed):
                    kg = float(st.session_state.get(f"cal_kg_{mk}_{i}", 0.0))
                    if kg <= cap + 0.5:
                        cfg.samples.append({"fill_ratio": round(meas.fill_ratio, 4), "known_kg": kg})
                        added += 1
                cv_bundle.set(active, cfg); cv_bundle.save(cfg_path)
                st.success(f"Added {added} samples to {active}.")
        with bcol2:
            if st.button(f"🗑️ Clear {active} samples", use_container_width=True):
                cfg.samples = []; cv_bundle.set(active, cfg); cv_bundle.save(cfg_path)
                st.warning(f"Cleared {active} samples.")

    # ============================================================
    #  Grab a calibration sample from a VIDEO frame
    # ============================================================
    theme.section("add sample from video")
    st.caption("Upload a clip of THIS feeder at a known weight, scrub to a clear frame, "
               "then add that frame as a calibration sample.")
    vid = st.file_uploader("Calibration video", type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
                           key=f"cal_vid_{mk}")
    if vid is not None and not previews_ok:
        st.info("ℹ️ This method's model isn't available yet — video preview disabled.")
    elif vid is not None:
        vframes = _frames_cached(vid.read(), 24)
        if not vframes:
            st.error("Could not read frames from this video.")
        else:
            vi = st.slider("Frame", 0, len(vframes) - 1, len(vframes) // 2, key=f"cal_vframe_{mk}")
            ts, jpg = vframes[vi]
            frame_img = Image.open(io.BytesIO(jpg))
            meas = _measure(frame_img, cv_bundle, active, _yolo_for(jpg, method))

            vc1, vc2 = st.columns(2)
            with vc1:
                st.image(meas.mask_image, use_container_width=True,
                         caption=f"t={ts:.1f}s · fill {meas.fill_ratio*100:.1f}%")
            with vc2:
                cap = logic.FEEDER_MAX_KG.get(active, 10.0)
                vkg = st.number_input("Known weight (kg) at this frame", 0.0, float(cap), 0.0, 0.1,
                                      key=f"cal_vkg_{mk}")
                st.markdown(f"fill ratio = `{meas.fill_ratio*100:.2f}%`")
                if st.button(f"➕ Add this frame to {active}", key=f"cal_vadd_{mk}",
                             use_container_width=True):
                    cfg.samples.append({"fill_ratio": round(meas.fill_ratio, 4), "known_kg": float(vkg)})
                    cv_bundle.set(active, cfg); cv_bundle.save(cfg_path)
                    st.success(f"Added frame (t={ts:.1f}s) → {vkg:.2f} kg to {active}.")

    # ============================================================
    #  Calibration table + curve
    # ============================================================
    theme.section(f"{active} calibration samples")
    if cfg.samples:
        st.dataframe(
            [{"#": i + 1, "fill_ratio_%": round(s["fill_ratio"] * 100, 2), "known_kg": s["known_kg"]}
             for i, s in enumerate(cfg.samples)],
            use_container_width=True, hide_index=True,
        )
        st.markdown(f"**Calibration:** piecewise interpolation (connect-the-dots) through "
                    f"{len(cfg.samples)} points — each known weight reads back exactly.")
        if not cv_feeder.calibration_monotonic(cfg.samples):
            st.warning("⚠️ fill-ratio is **not** strictly increasing with weight (a higher kg gave "
                       "a lower fill %). Interpolation may behave oddly between those points — "
                       "re-check / re-shoot the odd samples.")
        pts = [{"known_kg": s["known_kg"], "fill_%": s["fill_ratio"] * 100} for s in cfg.samples]
        st.scatter_chart(pts, x="known_kg", y="fill_%", height=220)
    else:
        st.info(f"No calibration samples for {active} yet.")
