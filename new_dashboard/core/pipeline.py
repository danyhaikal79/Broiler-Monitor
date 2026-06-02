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
def warm_models():
    inference.load_feeder_model()
    inference.load_chicken_model()
    return True


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_infer(img_hash: str, _img: Image.Image):
    """Cache YOLO feeder+chicken results by image content. Lets the page
    auto-refresh (e.g. live sensor every 5s) WITHOUT re-running the models on
    the same frame — only the cheap THI/feed math recomputes."""
    feeder = inference.run_feeder(_img, conf=FEEDER_CONF)
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
            override_feeder_type: str | None = None) -> Bundle:
    """Run YOLO feeder + CV feed + YOLO chicken + prediction. One call.
    YOLO is cached by image content; env changes only recompute the cheap math."""
    arr = np.asarray(img.convert("RGB"))
    img_hash = hashlib.md5(arr.tobytes()).hexdigest()
    feeder, chicken = _cached_infer(img_hash, img)

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
        chicken_count=chicken.count,
        current_food_kg=cv_meas.current_food_kg if cv_meas else 0.0,
        feeder_type=detected,
    )
    return Bundle(feeder=feeder, chicken=chicken, cv_meas=cv_meas, pred=pred,
                  detected_feeder_type=detected)
