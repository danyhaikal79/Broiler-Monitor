"""
Worker control channel — local files shared between the dashboard and the worker.

The dashboard and worker.py are SEPARATE processes on the same device (Jetson).
They coordinate through small files in the project root:

  monitor_control.json   dashboard -> worker : {"camera_enabled": bool}
                         Live mode sets it True  (worker captures + logs);
                         Upload mode sets it False (worker pauses its camera).
  monitor_latest.json    worker -> dashboard : most recent reading + epoch + the
                         annotated frame (base64 JPEG), all in ONE atomic write so
                         the numbers and the frame can never be a cycle out of sync.

The WORKER is the only thing that writes to the database. The dashboard only
flips the flag and displays what the worker publishes here.

Writes are atomic (temp file + os.replace) so a reader never sees a half-written
file. All files live next to monitor_config.json (config.ROOT).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time

from . import config

CONTROL_PATH = config.ROOT / "monitor_control.json"
LATEST_JSON_PATH = config.ROOT / "monitor_latest.json"
UPLOAD_REQUEST_PATH = config.ROOT / "monitor_upload_request.json"
UPLOAD_REQUEST_BIN_PATH = config.ROOT / "monitor_upload_request.bin"   # raw file bytes (sidecar)
UPLOAD_RESULT_PATH = config.ROOT / "monitor_upload_result.json"


def _atomic_write_bytes(path, data: bytes):
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)   # atomic on the same filesystem


def _atomic_write_json(path, obj):
    _atomic_write_bytes(path, json.dumps(obj).encode("utf-8"))


# ----------------------------------------------------- dashboard -> worker
def read_control() -> dict:
    """Worker reads this each poll. Defaults to paused if the file is missing/bad.
    env_override = optional {temp,hum,age} to use instead of the sensor/fallback."""
    try:
        with open(CONTROL_PATH, "r") as f:
            d = json.load(f)
        return {"camera_enabled": bool(d.get("camera_enabled", False)),
                "env_override": d.get("env_override")}
    except Exception:
        return {"camera_enabled": False, "env_override": None}


def set_control(camera_enabled: bool, env_override=None):
    """Dashboard -> worker. camera_enabled = Live(True)/Upload(False). env_override =
    a {temp,hum,age} dict to make the worker use those values instead of its DHT-22 /
    config fallback (handy for a live demo with no sensor); None = use sensor/fallback."""
    _atomic_write_json(CONTROL_PATH, {"camera_enabled": bool(camera_enabled),
                                      "env_override": env_override, "epoch": time.time()})


def is_fresh(age_s: float, interval: int) -> bool:
    """Single source of truth for 'is the worker's data fresh' across all UI
    surfaces (sidebar badge + live view), so they never contradict each other."""
    return age_s < max(30, int(interval) * 3)


# ----------------------------------------------------- worker -> dashboard
# The worker publishes a RICH result so the dashboard can render the full readout
# (Original / Feeder / Feed-mask / Chickens image tabs + status panel) WITHOUT doing
# any inference. 'images' = dict of PIL images; 'extras' = extra display numbers
# (wet-bulb, stress factor, detection confidences). Packed into ONE atomic write so
# numbers + frames are always in sync.
def _enc_img(pil):
    if pil is None:
        return None
    try:
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _pack(reading, coverage, images, extras):
    return {"reading": reading, "coverage": coverage, "epoch": time.time(),
            "extras": extras or {},
            "images_b64": {k: _enc_img(v) for k, v in (images or {}).items()}}


def _unpack(d):
    d["age_s"] = max(0.0, time.time() - float(d.get("epoch", 0)))
    d["extras"] = d.get("extras") or {}
    imgs = d.pop("images_b64", None) or {}
    out = {}
    for k, v in imgs.items():
        try:
            out[k] = base64.b64decode(v) if v else None
        except Exception:
            out[k] = None
    d["images"] = out
    return d


def write_latest(reading: dict, coverage: float, *, images=None, extras=None):
    """Worker -> dashboard: the latest LIVE result (reading + images + extras)."""
    _atomic_write_json(LATEST_JSON_PATH, _pack(reading, coverage, images, extras))


def read_latest():
    """Dashboard reads the worker's latest live publish, or None. Adds 'age_s',
    'images' (dict name->jpeg bytes|None) and 'extras' (dict)."""
    try:
        with open(LATEST_JSON_PATH, "r") as f:
            d = json.load(f)
    except Exception:
        return None
    return _unpack(d)


# ---------------------------------------------- upload handoff (dashboard <-> worker)
# The dashboard does NO inference. To analyze an uploaded image/video it drops the
# file here; the WORKER picks it up, runs YOLO + logs to the DB, and writes the
# result back. One id correlates request <-> result.
def submit_upload(req_id: str, kind: str, data_bytes: bytes, *, method, temp, hum, age, max_frames=20):
    """Dashboard -> worker: hand off an uploaded file. The raw bytes go to a binary
    sidecar (NOT base64-in-JSON) to avoid the +33% inflation and a giant JSON string
    in RAM on a constrained Jetson. The JSON carries only small metadata + the path.
    Write the bytes BEFORE the JSON so the JSON's presence implies the bytes are ready."""
    _atomic_write_bytes(UPLOAD_REQUEST_BIN_PATH, data_bytes)
    _atomic_write_json(UPLOAD_REQUEST_PATH, {
        "id": req_id, "kind": kind, "method": int(method),
        "temp": float(temp), "hum": float(hum), "age": int(age), "max_frames": int(max_frames),
        "data_path": str(UPLOAD_REQUEST_BIN_PATH), "epoch": time.time(),
    })


def read_upload_request():
    """Worker reads the pending upload request (bytes loaded from the sidecar), or None."""
    try:
        with open(UPLOAD_REQUEST_PATH, "r") as f:
            d = json.load(f)
        with open(d.get("data_path", str(UPLOAD_REQUEST_BIN_PATH)), "rb") as bf:
            d["data_bytes"] = bf.read()
        return d
    except Exception:
        return None


def consume_upload_request():
    """Worker: delete the request (json + sidecar) once processed, so a worker restart
    can't re-run a stale request and double-log it."""
    for p in (UPLOAD_REQUEST_PATH, UPLOAD_REQUEST_BIN_PATH):
        try:
            os.remove(p)
        except Exception:
            pass


def write_upload_result(req_id: str, reading, coverage, *, images=None, extras=None, error=None):
    """Worker -> dashboard: the analyzed result for an upload (correlated by req_id)."""
    p = _pack(reading, coverage, images, extras)
    p["id"] = req_id
    p["error"] = error
    _atomic_write_json(UPLOAD_RESULT_PATH, p)


def read_upload_result():
    """Dashboard reads the most recent upload result, or None. Adds 'age_s','images','extras'."""
    try:
        with open(UPLOAD_RESULT_PATH, "r") as f:
            d = json.load(f)
    except Exception:
        return None
    return _unpack(d)
