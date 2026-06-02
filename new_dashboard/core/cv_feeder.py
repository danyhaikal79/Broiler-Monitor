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

CONFIG_PATH = Path(__file__).resolve().parent / "cv_config.json"
FEEDER_TYPES = ("pan3kg", "pan7kg", "tube7kg")


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
    """All three per-feeder configs stored together."""
    pan3kg: FeederCVConfig = field(default_factory=FeederCVConfig)
    pan7kg: FeederCVConfig = field(default_factory=FeederCVConfig)
    tube7kg: FeederCVConfig = field(default_factory=FeederCVConfig)

    def get(self, feeder_type: str) -> FeederCVConfig:
        return getattr(self, feeder_type, self.pan3kg)

    def set(self, feeder_type: str, cfg: FeederCVConfig) -> None:
        setattr(self, feeder_type, cfg)

    def save(self, path: Path = CONFIG_PATH) -> None:
        data = {ft: getattr(self, ft).__dict__ for ft in FEEDER_TYPES}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "CVConfigBundle":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()

        # Backward-compat: old single-config format -> put into pan3kg slot
        if "roi_x1_pct" in data:
            bundle = cls()
            old = FeederCVConfig(**{k: v for k, v in data.items()
                                    if k in FeederCVConfig.__annotations__})
            ft = data.get("feeder_type", "pan3kg")
            if ft in FEEDER_TYPES:
                bundle.set(ft, old)
            return bundle

        # New multi-config format
        bundle = cls()
        for ft in FEEDER_TYPES:
            if ft in data:
                bundle.set(ft, FeederCVConfig(**data[ft]))
        return bundle


# ----------------------------------------------------------------------------
# Calibration: fit a line through (fill_ratio, known_kg) sample points
# ----------------------------------------------------------------------------
def fit_calibration(samples: list[dict]) -> tuple[float, float, float]:
    """Fit weight = slope * fill_ratio + intercept. Returns (slope, intercept, R²)."""
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
    slope, intercept, _ = fit_calibration(cfg.samples)
    current_kg = max(0.0, slope * fill_ratio + intercept)

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
