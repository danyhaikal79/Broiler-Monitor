"""
YOLO model wrappers for the dashboard.

Two models:
- Feeder segmentation (YOLOv11n-seg) -> current feed weight + feeder type
- Chicken detection (YOLO12n)         -> chicken count

Both models are loaded once (cached) and run on a single PIL Image.
Returns plain dicts so logic.py / app.py don't need to import ultralytics.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

def _find_project_root() -> Path:
    """Walk up until we find the fyp3 root (the folder holding feeder_train + chicken_train).
    Robust to however deeply this module is nested under new_dashboard/."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "feeder_train").is_dir() and (parent / "chicken_train").is_dir():
            return parent
    return here.parent.parent  # fallback

ROOT = _find_project_root()

# Paths to the deployed models. Feeder segmentation models (3-class:
# pan3kg/pan7kg/tube7kg); feed level is done by CV inside the chosen polygon:
#   Method 1 = WHOLE feeder polygon   Method 2 = OPEN-AREA polygon
#   Method 3 = DEMO (dummy farm) — own model once trained, else uses Method 2's for now.
FEEDER_WEIGHTS = ROOT / "feeder_train" / "runs" / "seg_compare" / "yolov8n" / "weights" / "best.pt"
FEEDER_WEIGHTS_M2 = ROOT / "feeder_train" / "runs" / "method2" / "weights" / "best.pt"
FEEDER_WEIGHTS_M3 = ROOT / "feeder_train" / "runs" / "method3" / "weights" / "best.pt"   # not trained yet
CHICKEN_WEIGHTS = ROOT / "chicken_train" / "runs" / "chicken_compare" / "yolo11n" / "weights" / "best.pt"

FEEDER_WEIGHTS_BY_METHOD = {
    1: FEEDER_WEIGHTS,
    2: FEEDER_WEIGHTS_M2,
    # Demo: use its own model once feeder_train/runs/method3/weights/best.pt exists,
    # otherwise fall back to Method 2's model for now.
    3: FEEDER_WEIGHTS_M3 if FEEDER_WEIGHTS_M3.exists() else FEEDER_WEIGHTS_M2,
}

# Shared UI labels for the feeder-estimation methods.
METHOD_LABELS = {1: "Method 1 · whole feeder", 2: "Method 2 · open area", 3: "Method 3 · demo"}

# Feeder class IDs (new 3-class model, from new_feeder_dataset):
#   0: pan3kg, 1: pan7kg, 2: tube7kg  (no 'feed' class -- CV handles feed level)
FEEDER_CLASS_NAMES = {0: "pan3kg", 1: "pan7kg", 2: "tube7kg"}
FEEDER_TYPES = {"pan3kg", "pan7kg", "tube7kg"}


# ----------------------------------------------------------------------------
# Lazy model loading -- import ultralytics only when needed (it's slow to load)
# ----------------------------------------------------------------------------
_feeder_models = {}   # method (1|2) -> loaded YOLO model (cached)
_chicken_model = None

# Serialize YOLO inference: the background live monitor (monitor.py) runs predict
# in a thread while the dashboard may also predict on the main thread. Ultralytics
# models are not safe for concurrent predict on the same object, so guard both.
_PREDICT_LOCK = threading.Lock()


def feeder_method_available(method: int = 1) -> bool:
    """True if the weights for this method exist (Method 2 may be untrained)."""
    return FEEDER_WEIGHTS_BY_METHOD.get(method, FEEDER_WEIGHTS).exists()


def load_feeder_model(method: int = 1):
    """Lazy-load + cache the feeder model for the chosen method. Cached by WEIGHTS PATH
    so methods sharing a model (e.g. Method 3 → Method 2's weights) load it only once."""
    global _feeder_models
    weights = FEEDER_WEIGHTS_BY_METHOD.get(method, FEEDER_WEIGHTS)
    key = str(weights)
    if key not in _feeder_models:
        from ultralytics import YOLO
        if not weights.exists():
            raise FileNotFoundError(
                f"Method {method} feeder weights not found: {weights} — train the model "
                "and place best.pt there (Method 3 falls back to Method 2's model until then).")
        _feeder_models[key] = YOLO(key)
    return _feeder_models[key]


def load_chicken_model():
    global _chicken_model
    if _chicken_model is None:
        from ultralytics import YOLO
        if not CHICKEN_WEIGHTS.exists():
            raise FileNotFoundError(f"Chicken weights not found: {CHICKEN_WEIGHTS}")
        _chicken_model = YOLO(str(CHICKEN_WEIGHTS))
    return _chicken_model


# ----------------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------------
@dataclass
class FeederResult:
    feeder_type: str | None        # "pan3kg" / "pan7kg" / "tube7kg" / None
    feeder_type_conf: float        # YOLO confidence of the chosen feeder type
    feeder_bbox: tuple | None      # (x1, y1, x2, y2) pixel coords of feeder body
    feeder_polygon: Any = None     # Nx2 np.ndarray of pixel-space polygon vertices
    fill_ratio: float = 0.0        # feed_area / pan_area, in [0, 1]
    current_food_kg: float = 0.0   # fill_ratio * max_capacity_for_feeder_type
    feed_area_px: float = 0.0      # raw pixel-area of feed mask(s) (sum)
    pan_area_px: float = 0.0       # raw pixel-area of feeder body mask
    annotated_image: Any = None    # PIL Image with masks/boxes overlaid


@dataclass
class ChickenResult:
    count: int
    avg_confidence: float
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    annotated_image: Any = None


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def _polygon_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Shoelace formula for polygon area in pixel units."""
    return 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def apply_clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid: int = 8) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization to even out lighting.
    Operates on the L channel of LAB so colors aren't shifted, only brightness
    contrast. Helps the model on images with uneven exposure / shadows that
    differ from training data lighting.
    """
    import cv2
    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb2)


def run_feeder(image: Image.Image, conf: float = 0.25, method: int = 1) -> FeederResult:
    """
    Run the feeder segmentation model (3-class: pan3kg/pan7kg/tube7kg).

    method 1 = whole-feeder polygon, method 2 = open-area polygon. Both return
    the feeder type + polygon/bbox used as the CV ROI; feed level is measured by
    the classical-CV pipeline (cv_feeder.py), NOT here.
    """
    model = load_feeder_model(method)
    arr = np.array(image.convert("RGB"))
    with _PREDICT_LOCK:
        results = model.predict(arr, conf=conf, verbose=False)
    res = results[0]

    pan_area = 0.0
    detected_feeder = None
    best_feeder_conf = 0.0
    detected_bbox = None       # (x1,y1,x2,y2) of best-conf feeder body
    detected_polygon = None    # Nx2 array of polygon vertices for best-conf feeder body

    if res.masks is not None and res.boxes is not None:
        polys = res.masks.xy
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        xyxys = res.boxes.xyxy.cpu().numpy()

        for poly, cls_id, c, xyxy in zip(polys, cls_ids, confs, xyxys):
            if len(poly) < 3:
                continue
            name = FEEDER_CLASS_NAMES.get(int(cls_id), "?")
            if name in FEEDER_TYPES and c > best_feeder_conf:
                best_feeder_conf = float(c)
                detected_feeder = name
                detected_bbox = tuple(float(v) for v in xyxy)
                detected_polygon = poly.copy()
                pan_area = _polygon_area(poly[:, 0], poly[:, 1])

    # Render annotated image (ultralytics returns BGR; ask for PIL/RGB directly)
    annotated = None
    try:
        annotated = res.plot(pil=True)
    except Exception:
        try:
            annotated = Image.fromarray(res.plot()[..., ::-1])
        except Exception:
            pass

    return FeederResult(
        feeder_type=detected_feeder,
        feeder_type_conf=best_feeder_conf,
        feeder_bbox=detected_bbox,
        feeder_polygon=detected_polygon,
        fill_ratio=0.0,          # feed level now handled by CV, not YOLO
        current_food_kg=0.0,     # "
        feed_area_px=0.0,        # "
        pan_area_px=pan_area,
        annotated_image=annotated,
    )


def run_chicken(image: Image.Image, conf: float = 0.25) -> ChickenResult:
    """Run the chicken detection model. Returns count + annotated preview."""
    model = load_chicken_model()
    arr = np.array(image.convert("RGB"))
    with _PREDICT_LOCK:
        results = model.predict(arr, conf=conf, verbose=False)
    res = results[0]

    count = 0
    avg_conf = 0.0
    boxes: list[tuple[float, float, float, float]] = []

    if res.boxes is not None and len(res.boxes) > 0:
        count = len(res.boxes)
        confs = res.boxes.conf.cpu().numpy()
        avg_conf = float(confs.mean())
        xyxy = res.boxes.xyxy.cpu().numpy()
        boxes = [tuple(map(float, b)) for b in xyxy]

    annotated = None
    try:
        annotated = res.plot(pil=True)
    except Exception:
        try:
            annotated = Image.fromarray(res.plot()[..., ::-1])
        except Exception:
            pass

    return ChickenResult(count=count, avg_confidence=avg_conf, boxes=boxes, annotated_image=annotated)
