"""
Classical-CV feed level estimator with per-feeder-type configs.

For a fixed-camera setup, this is more reliable than YOLO segmentation:
- No training data needed for the feed-level math
- Deterministic, debuggable
- Runs in <5 ms on Jetson Nano
- Trade-off: needs manual calibration per camera / feeder setup

A separate config (ROI + HSV + calibration curve) is stored for each feeder
type (pan3kg, pan7kg, tube7kg). The active config is picked by either:
- Auto-detect via the YOLO feeder model (used only for classification), or
- Manual override from the dashboard UI

Pipeline (per image):
  1. Identify feeder type (auto or manual)
  2. Load the matching per-feeder config
  3. Crop input image to that config's ROI
  4. Convert ROI to HSV color space
  5. Threshold for the feed color range
  6. Count feed pixels inside the ROI
  7. Apply that feeder's calibration curve (fill_ratio -> kg)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

_CORE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _CORE_DIR / "cv_config.json"                 # Method 1 (whole feeder)
CONFIG_PATH_METHOD2 = _CORE_DIR / "cv_config_method2.json"  # Method 2 (open area)
CONFIG_PATH_METHOD3 = _CORE_DIR / "cv_config_method3.json"  # Method 3 (demo / dummy farm)
FEEDER_TYPES = ("pan3kg", "pan7kg", "tube7kg", "feeder-demo-3kg")

# Which feeder types belong to each method. Methods 1/2 are the real feeders; Method 3
# is the demo, whose ONLY feeder is the dummy cup. Keeping these separate stops a
# method's config file from accumulating types it never uses, and stops cross-method
# pollution (e.g. calibrating the demo while on Method 1 landing in Method 1's file).
FEEDER_TYPES_BY_METHOD = {
    1: ("pan3kg", "pan7kg", "tube7kg"),
    2: ("pan3kg", "pan7kg", "tube7kg"),
    3: ("feeder-demo-3kg",),
}


def config_path_for(method: int = 1) -> Path:
    """Which CV-config file backs a given method (each is calibrated separately)."""
    if method == 2:
        return CONFIG_PATH_METHOD2
    if method == 3:
        return CONFIG_PATH_METHOD3
    return CONFIG_PATH


def feeder_types_for(method: int | None = None) -> tuple:
    """Feeder types relevant to a method (all of them if method is None)."""
    if method is None:
        return FEEDER_TYPES
    return FEEDER_TYPES_BY_METHOD.get(int(method), FEEDER_TYPES)


# ----------------------------------------------------------------------------
# Per-feeder config + bundle (all three feeder types in one file)
# ----------------------------------------------------------------------------
@dataclass
class FeederCVConfig:
    """CV config for ONE feeder type."""
    # ROI as percentages of full image (0..100)
    roi_x1_pct: float = 30.0
    roi_y1_pct: float = 30.0
    roi_x2_pct: float = 70.0
    roi_y2_pct: float = 70.0

    # Detection method: "hsv", "texture", or "combined".
    # - "hsv": color threshold. Use when feed color differs from feeder (pans).
    # - "texture": granularity. Use when feed = feeder color but feed is rougher (tube).
    # - "combined": pixel must pass BOTH HSV and texture (most robust, more tuning).
    detect_method: str = "hsv"

    # HSV thresholds for "feed" (defaults: yellow-brown broiler feed)
    hsv_h_min: int = 10
    hsv_h_max: int = 35
    hsv_s_min: int = 40
    hsv_s_max: int = 255
    hsv_v_min: int = 50
    hsv_v_max: int = 255

    # Texture-detection params (used when detect_method == "texture")
    texture_threshold: float = 12.0   # local std-dev cutoff: higher = only rougher regions
    texture_window: int = 7           # sliding-window size (odd number)

    # Calibration samples: each {fill_ratio, known_kg}
    samples: list[dict] = field(default_factory=list)


@dataclass
class CVConfigBundle:
    """Per-feeder CV configs, keyed by feeder-type NAME. A dict (not fixed fields) so it
    accepts any type — including the hyphenated demo type 'feeder-demo-3kg'."""
    configs: dict = field(default_factory=dict)

    def get(self, feeder_type: str) -> FeederCVConfig:
        return self.configs.get(feeder_type) or FeederCVConfig()

    def set(self, feeder_type: str, cfg: FeederCVConfig) -> None:
        self.configs[feeder_type] = cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        data = {ft: c.__dict__ for ft, c in self.configs.items()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH, method: int | None = None) -> "CVConfigBundle":
        """Load the per-feeder configs from `path`. If `method` is given, only that
        method's feeder types are kept/created (see FEEDER_TYPES_BY_METHOD) — so e.g.
        Method 3's file holds ONLY feeder-demo-3kg, and a stray entry from another method
        is dropped on the next save instead of accumulating. method=None keeps all types."""
        bundle = cls()
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}

        allowed = feeder_types_for(method)
        if "roi_x1_pct" in data:                       # backward-compat: old single-config format
            ft = data.get("feeder_type", "pan3kg")
            if ft in allowed:
                bundle.configs[ft] = FeederCVConfig(**{k: v for k, v in data.items()
                                                       if k in FeederCVConfig.__annotations__})
        else:                                          # multi-config: load every RELEVANT key
            for ft, c in data.items():
                if ft not in allowed:                  # skip types that don't belong to this method
                    continue
                try:
                    bundle.configs[ft] = FeederCVConfig(**c)
                except Exception:
                    pass

        for ft in allowed:                             # ensure this method's types are editable
            bundle.configs.setdefault(ft, FeederCVConfig())
        return bundle


# ----------------------------------------------------------------------------
# Calibration: fit a line through (fill_ratio, known_kg) sample points
# ----------------------------------------------------------------------------
def predict_kg(samples: list[dict], fill_ratio: float) -> float:
    """
    Convert a fill ratio -> kg by PIECEWISE-LINEAR interpolation through the
    calibration points (connect-the-dots). Unlike a single straight-line fit,
    this passes EXACTLY through every sample, so it doesn't overshoot bumpy data.
    Values outside the sampled range are clamped to the nearest endpoint.
    """
    pts = [(float(s["fill_ratio"]), float(s["known_kg"]))
           for s in samples if s.get("fill_ratio") is not None]
    if not pts:
        return max(0.0, fill_ratio)                       # no calibration -> raw ratio
    if len(pts) == 1:
        fr, kg = pts[0]
        return max(0.0, (kg / fr) * fill_ratio) if fr > 0 else max(0.0, kg)
    pts.sort(key=lambda p: p[0])                           # sort by fill_ratio
    xs, ys = [], []
    for fr, kg in pts:                                     # collapse duplicate x (avg y)
        if xs and abs(fr - xs[-1]) < 1e-9:
            ys[-1] = (ys[-1] + kg) / 2.0
        else:
            xs.append(fr); ys.append(kg)
    return max(0.0, float(np.interp(fill_ratio, xs, ys)))  # np.interp clamps at the ends


def calibration_monotonic(samples: list[dict]) -> bool:
    """True if fill_ratio strictly increases with known_kg — the condition for
    piecewise interpolation to behave sensibly. Non-monotonic data means a
    higher weight gave a LOWER fill ratio (a bad/noisy sample)."""
    pts = sorted((float(s["known_kg"]), float(s["fill_ratio"]))
                 for s in samples if s.get("fill_ratio") is not None)
    fills = [f for _, f in pts]
    return all(b > a for a, b in zip(fills, fills[1:])) if len(fills) > 1 else True


def fit_calibration(samples: list[dict]) -> tuple[float, float, float]:
    """Linear fit weight = slope*fill + intercept; returns (slope, intercept, R²).
    Kept for an informational 'how linear is the data' read-out only — the actual
    kg prediction uses predict_kg (piecewise interpolation)."""
    if len(samples) < 2:
        if len(samples) == 1 and samples[0]["fill_ratio"] > 0:
            s = samples[0]
            return s["known_kg"] / s["fill_ratio"], 0.0, 1.0
        return 1.0, 0.0, 0.0

    x = np.array([s["fill_ratio"] for s in samples], dtype=float)
    y = np.array([s["known_kg"] for s in samples], dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------
@dataclass
class CVMeasurement:
    feeder_type: str
    fill_ratio: float
    current_food_kg: float
    feed_pixels: int
    roi_pixels: int
    detected_type_source: str = "unknown"   # "yolo" / "manual" / "default"
    detected_confidence: float = 0.0
    mask_image: Image.Image | None = None
    roi_image: Image.Image | None = None
    full_overlay: Image.Image | None = None


def _texture_mask(rgb: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """
    Binary mask of high-texture (granular) regions via local standard deviation.
    Feed pellets are bumpy -> high local std. Smooth plastic (cone/pan) -> low std.
    Independent of color, so it separates same-colored-but-different-roughness things.
    """
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = window if window % 2 == 1 else window + 1
    mean = cv2.boxFilter(gray, -1, (k, k))
    mean_sq = cv2.boxFilter(gray * gray, -1, (k, k))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    return (std > threshold).astype(np.uint8) * 255


def _crop_roi(image: Image.Image, cfg: FeederCVConfig) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = image.size
    x1 = int(w * min(cfg.roi_x1_pct, cfg.roi_x2_pct) / 100)
    x2 = int(w * max(cfg.roi_x1_pct, cfg.roi_x2_pct) / 100)
    y1 = int(h * min(cfg.roi_y1_pct, cfg.roi_y2_pct) / 100)
    y2 = int(h * max(cfg.roi_y1_pct, cfg.roi_y2_pct) / 100)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return image, (0, 0, w, h)
    return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def measure(
    image: Image.Image,
    bundle: CVConfigBundle,
    feeder_type: str,
    detected_source: str = "manual",
    detected_conf: float = 0.0,
    yolo_bbox: tuple | None = None,
    yolo_polygon=None,
    mask_dilation_px: int = 0,
) -> CVMeasurement:
    """
    Run the full CV pipeline on one image using the config for `feeder_type`.

    ROI selection priority:
      1. `yolo_polygon` (Nx2 vertex array) -> precise polygon mask. Best accuracy.
      2. `yolo_bbox` (x1,y1,x2,y2) -> rectangular bbox.
      3. Saved ROI from `bundle.get(feeder_type)`.

    `mask_dilation_px` expands the polygon mask outward (handles slight
    under-segmentation by YOLO).
    """
    import cv2
    from PIL import ImageDraw

    cfg = bundle.get(feeder_type)
    full_w, full_h = image.size
    poly_mask = None    # binary mask within ROI; only used when polygon provided

    # ---- Decide the ROI rectangle (+ polygon mask if available) ----
    if yolo_polygon is not None and len(yolo_polygon) >= 3:
        xs = yolo_polygon[:, 0]
        ys = yolo_polygon[:, 1]
        x1 = max(0, int(xs.min()))
        y1 = max(0, int(ys.min()))
        x2 = min(full_w, int(xs.max()))
        y2 = min(full_h, int(ys.max()))
        if x2 - x1 < 10 or y2 - y1 < 10:
            roi_img, (x1, y1, x2, y2) = _crop_roi(image, cfg)
        else:
            roi_img = image.crop((x1, y1, x2, y2))
            poly_local = (yolo_polygon - np.array([x1, y1])).astype(np.int32)
            poly_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [poly_local], 255)
            if mask_dilation_px > 0:
                kd = np.ones((mask_dilation_px * 2 + 1,) * 2, np.uint8)
                poly_mask = cv2.dilate(poly_mask, kd)
    elif yolo_bbox is not None:
        bx1, by1, bx2, by2 = yolo_bbox
        x1 = max(0, int(bx1))
        y1 = max(0, int(by1))
        x2 = min(full_w, int(bx2))
        y2 = min(full_h, int(by2))
        if x2 - x1 < 10 or y2 - y1 < 10:
            roi_img, (x1, y1, x2, y2) = _crop_roi(image, cfg)
        else:
            roi_img = image.crop((x1, y1, x2, y2))
    else:
        roi_img, (x1, y1, x2, y2) = _crop_roi(image, cfg)

    # ---- Build the feed mask via the configured detection method ----
    rgb = np.array(roi_img.convert("RGB"))

    def _hsv_mask():
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([cfg.hsv_h_min, cfg.hsv_s_min, cfg.hsv_v_min], dtype=np.uint8)
        upper = np.array([cfg.hsv_h_max, cfg.hsv_s_max, cfg.hsv_v_max], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    if cfg.detect_method == "texture":
        feed_mask = _texture_mask(rgb, cfg.texture_threshold, cfg.texture_window)
    elif cfg.detect_method == "combined":
        # pixel must be BOTH feed-colored AND granular
        feed_mask = cv2.bitwise_and(
            _hsv_mask(),
            _texture_mask(rgb, cfg.texture_threshold, cfg.texture_window),
        )
    else:  # "hsv"
        feed_mask = _hsv_mask()

    # ---- Restrict feed detection to polygon (if available) ----
    if poly_mask is not None:
        feed_mask = cv2.bitwise_and(feed_mask, poly_mask)

    # ---- Morphological cleanup ----
    k = np.ones((3, 3), np.uint8)
    feed_mask = cv2.morphologyEx(feed_mask, cv2.MORPH_OPEN, k)
    feed_mask = cv2.morphologyEx(feed_mask, cv2.MORPH_CLOSE, k)

    # ---- Count + fill ratio ----
    feed_pixels = int((feed_mask > 0).sum())
    if poly_mask is not None:
        roi_pixels = int((poly_mask > 0).sum())   # only pixels INSIDE polygon
    else:
        roi_pixels = int(feed_mask.size)
    fill_ratio = (feed_pixels / roi_pixels) if roi_pixels > 0 else 0.0

    # ---- Calibration ----
    current_kg = predict_kg(cfg.samples, fill_ratio)   # piecewise (connect-the-dots)

    # ---- Mask preview overlay ----
    overlay = rgb.copy()
    overlay[feed_mask > 0] = (
        0.5 * overlay[feed_mask > 0] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)
    if poly_mask is not None:
        contours, _ = cv2.findContours(poly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    mask_img = Image.fromarray(overlay)

    # ---- Full-image preview ----
    full_preview = image.copy()
    draw = ImageDraw.Draw(full_preview)
    if yolo_polygon is not None and len(yolo_polygon) >= 3:
        pts = [(float(p[0]), float(p[1])) for p in yolo_polygon]
        draw.line(pts + [pts[0]], fill=(0, 255, 0), width=4)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 215, 0), width=6)

    return CVMeasurement(
        feeder_type=feeder_type,
        fill_ratio=fill_ratio,
        current_food_kg=current_kg,
        feed_pixels=feed_pixels,
        roi_pixels=roi_pixels,
        detected_type_source=detected_source,
        detected_confidence=detected_conf,
        mask_image=mask_img,
        roi_image=roi_img,
        full_overlay=full_preview,
    )


# ----------------------------------------------------------------------------
# Auto-detect feeder type from YOLO feeder model result
# ----------------------------------------------------------------------------
def detect_feeder_type_from_yolo(feeder_result, fallback: str = "pan3kg") -> tuple[str, float]:
    """
    Take a FeederResult from inference.run_feeder() and return (feeder_type, confidence).

    Only uses YOLO to CLASSIFY (not to measure feed). Lower confidence threshold
    than usual since we just need to know which type, not segment precisely.
    """
    detected = getattr(feeder_result, "feeder_type", None)
    if detected and detected in FEEDER_TYPES:
        # Confidence isn't carried back from FeederResult, just say "found"
        return detected, 1.0
    return fallback, 0.0
