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
import time

import requests
from PIL import Image

from . import db

# Cache the auto-discovered worker URL so we don't query Supabase on every ping/frame.
_DISCOVER = {"url": None, "t": 0.0}
_DISCOVER_TTL = 15.0   # re-check the worker's published IP at most this often (seconds)


def base_url(cfg) -> str:
    """The worker's base URL.

    - worker_host = a real hostname/IP  -> use it directly (manual override).
    - worker_host = 'auto' (default)    -> discover the worker's CURRENT ip:port from
      Supabase (it publishes there every ~10s), cached. No mDNS, no hand-typed IP,
      survives the Jetson's IP changing on a hotspot."""
    host = (cfg.get("worker_host") or "auto").strip()
    port = int(cfg.get("worker_http_port", 8077))
    if host.lower() not in ("auto", ""):
        return f"http://{host}:{port}"

    now = time.time()
    if _DISCOVER["url"] and (now - _DISCOVER["t"] < _DISCOVER_TTL):
        return _DISCOVER["url"]
    status = db.fetch_worker_status(cfg)
    if status and status.get("ip"):
        _DISCOVER["url"] = f"http://{status['ip']}:{int(status.get('port') or port)}"
        _DISCOVER["t"] = now
        return _DISCOVER["url"]
    # nothing published yet -> keep last known if any, else an address that just fails fast
    return _DISCOVER["url"] or f"http://0.0.0.0:{port}"


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


def fetch_env(cfg, timeout: float = 10.0):
    """The Jetson's current sensor reading {temp, hum, age}, or None if unreachable.
    Reuses /frame (the reading rides in its headers) so no extra worker endpoint is
    needed; the image is discarded. Used by the dashboard's 'Auto' environment option."""
    img, meta = fetch_frame(cfg, timeout=timeout)
    return meta if img is not None else None
