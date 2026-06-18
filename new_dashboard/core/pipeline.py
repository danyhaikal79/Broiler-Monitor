"""
Shared inference + prediction pipeline. Both the finalise and calibrate views
call run_all() so the core logic lives in exactly one place.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import streamlit as st
from PIL import Image

from . import inference, cv_feeder, logic

FEEDER_CONF = 0.25
CHICKEN_CONF = 0.25


@st.cache_resource(show_spinner="Loading models (first run only)...")
def warm_models(method: int = 1):
    inference.load_feeder_model(method)
    inference.load_chicken_model()
    return True


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_infer(img_hash: str, method: int, _img: Image.Image):
    """Cache YOLO feeder+chicken results by image content AND method, so the page
    auto-refresh (e.g. live sensor every 5s) doesn't re-run the models on the same
    frame — only the cheap THI/feed math recomputes. Keyed on method so switching
    Method 1<->2 re-runs correctly."""
    feeder = inference.run_feeder(_img, conf=FEEDER_CONF, method=method)
    chicken = inference.run_chicken(_img, conf=CHICKEN_CONF)
    return feeder, chicken


@dataclass
class Bundle:
    feeder: object
    chicken: object
    cv_meas: object
    pred: object
    detected_feeder_type: str | None


def run_all(img: Image.Image, cv_bundle, *, temperature_c, humidity_pct, age_days,
            override_feeder_type: str | None = None, method: int = 1,
            chicken_count_override=None) -> Bundle:
    """Run YOLO feeder + CV feed + YOLO chicken + prediction. One call.
    `method` (1=whole feeder, 2=open area) selects which feeder model + ROI.
    `chicken_count_override` (from config) wins over the YOLO count, matching
    worker.py so the upload preview agrees with what the worker would log.
    YOLO is cached by image content + method; env changes only recompute cheap math."""
    arr = np.asarray(img.convert("RGB"))
    img_hash = hashlib.md5(arr.tobytes()).hexdigest()
    feeder, chicken = _cached_infer(img_hash, method, img)

    detected = override_feeder_type or feeder.feeder_type
    cv_meas = None
    if detected:
        cv_meas = cv_feeder.measure(
            img, cv_bundle, detected,
            detected_source=("manual" if override_feeder_type else "yolo"),
            detected_conf=feeder.feeder_type_conf,
            yolo_polygon=feeder.feeder_polygon, yolo_bbox=feeder.feeder_bbox,
            mask_dilation_px=0,
        )

    pred = logic.compute(
        temperature_c=temperature_c, humidity_pct=humidity_pct, age_days=age_days,
        chicken_count=(chicken_count_override if chicken_count_override is not None else chicken.count),
        current_food_kg=cv_meas.current_food_kg if cv_meas else 0.0,
        feeder_type=detected,
    )
    return Bundle(feeder=feeder, chicken=chicken, cv_meas=cv_meas, pred=pred,
                  detected_feeder_type=detected)
