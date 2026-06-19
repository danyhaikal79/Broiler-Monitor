"""
Dashboard-side client for the worker's LAN HTTP server (split deployment).

The worker (Jetson) exposes a tiny HTTP server (stdlib, no extra deps):
  GET /ping   -> presence heartbeat; also tells the worker "the dashboard is here"
  GET /frame  -> ONE fresh camera JPEG, with X-Temp / X-Hum / X-Age headers
                 (the Jetson's current sensor reading travels with the frame)
  GET /health -> worker status JSON

The dashboard (laptop) calls these over the local network, addressing the Jetson
by its mDNS HOSTNAME (config `worker_host`, e.g. jetson00-desktop.local) so a
changing IP on a hotspot never matters. These are server-to-server `requests`
calls (no browser CORS involved).
"""

from __future__ import annotations

import io

import requests
from PIL import Image


def base_url(cfg) -> str:
    host = cfg.get("worker_host", "jetson00-desktop.local")
    port = int(cfg.get("worker_http_port", 8077))
    return f"http://{host}:{port}"


def ping_url(url: str, timeout: float = 2.5) -> bool:
    """True if the worker at `url` answered /ping. Never raises."""
    try:
        return requests.get(url + "/ping", timeout=timeout).status_code == 200
    except Exception:
        return False


def ping(cfg, timeout: float = 3.0) -> bool:
    """True if the worker answered. Also marks the dashboard as 'present' on the worker."""
    return ping_url(base_url(cfg), timeout)


def fetch_frame(cfg, timeout: float = 10.0):
    """Pull one live frame from the worker.

    Returns (PIL.Image, env_dict) on success, where env_dict = {temp, hum, age}
    carries the Jetson's current sensor reading. On failure returns (None, error_str)."""
    url = base_url(cfg)
    try:
        r = requests.get(url + "/frame", timeout=timeout)
    except requests.exceptions.RequestException as e:
        return None, (f"can't reach the worker at {url} — is worker.py running on the Jetson, "
                      f"and are you on the same Wi-Fi? ({e})")
    if r.status_code == 503:
        return None, "worker reached, but its camera is unavailable (busy / unplugged)."
    if r.status_code != 200:
        return None, f"worker returned HTTP {r.status_code}"
    try:
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        return None, f"bad frame from worker: {e}"

    def _hf(name, default):
        try:
            return float(r.headers.get(name, default))
        except Exception:
            return default

    env = {"temp": _hf("X-Temp", 25.0), "hum": _hf("X-Hum", 70.0), "age": int(_hf("X-Age", 21))}
    return img, env
