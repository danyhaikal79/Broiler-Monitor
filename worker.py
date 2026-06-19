"""
Background monitoring worker — runs ON THE JETSON, separate from the dashboard.

Split deployment (laptop dashboard + Jetson worker, same Wi-Fi):

  • The worker owns the camera + DHT-22 sensor.
  • It serves a tiny HTTP API on `worker_http_port` (stdlib, no extra deps):
        GET /ping   -> "the dashboard is here" heartbeat
        GET /frame  -> one fresh camera JPEG (+ X-Temp/X-Hum/X-Age headers)
        GET /health -> status JSON
  • PRESENCE RULE — the only time the worker does inference + writes the DB itself:
        - dashboard ABSENT (no /ping or /frame within `presence_timeout_sec`):
              the worker captures every `interval_sec`, runs YOLO + CV, logs the
              reading, and fires the low-feed + heat Telegram alerts.  ← 24/7 fallback
        - dashboard PRESENT (laptop open & pinging):
              the worker STANDS DOWN — no inference, no DB writes. It only serves
              /frame on request; the laptop dashboard does the inference + logging.
  • Two-way Telegram (/status, /latest, /capture, /help) works in either state.

Only ONE process may own the camera — don't run two workers.

Run (on the Jetson):
  python3 worker.py
"""

import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "new_dashboard"))

from PIL import Image  # noqa: E402
from core import config, db, notify, inference, cv_feeder, sensor, video, engine  # noqa: E402


def _read_env(cfg):
    """(temp, hum): from the DHT-22 if use_sensor, else the config fallback."""
    if cfg.get("use_sensor"):
        reading, err = sensor.read_latest(cfg.get("serial_port", "/dev/ttyUSB0"),
                                          int(cfg.get("serial_baud", 115200)))
        if reading:
            return reading[0], reading[1]
        print(f"[worker] sensor read failed ({err}); using fallback")
    return float(cfg.get("temperature_fallback_c", 25.0)), float(cfg.get("humidity_fallback_pct", 70.0))


# ----------------------------------------------------------------------------
# Shared state between the main loop and the HTTP server threads
# ----------------------------------------------------------------------------
class Shared:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cam_lock = threading.Lock()        # serialise camera access (loop vs /frame)
        self._contact_lock = threading.Lock()
        self.last_contact_t = 0.0               # last time the dashboard pinged/fetched
        self.latest_env = (float(cfg.get("temperature_fallback_c", 25.0)),
                           float(cfg.get("humidity_fallback_pct", 70.0)))
        self._cap = None                        # PERSISTENT camera handle (see capture_pil)

    def mark_contact(self):
        with self._contact_lock:
            self.last_contact_t = time.time()

    def seconds_since_contact(self):
        with self._contact_lock:
            return time.time() - self.last_contact_t

    # --- Camera: open ONCE and keep it open. Re-opening per frame costs ~15-20s on a
    #     Jetson Nano; reusing the handle makes each grab near-instant, so live mode is
    #     smooth. Only the FIRST open pays the warm-up cost.
    def _open_cam(self):
        import cv2
        import platform
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(self.cfg.get("camera_index", 0)), backend)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(3):      # discard warm-up frames (only on open, NOT every grab)
            cap.read()
        return cap

    def _release_cam(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def capture_pil(self):
        """One frame from the PERSISTENT handle. Returns a PIL.Image or None."""
        import cv2
        with self.cam_lock:
            if self._cap is None or not self._cap.isOpened():
                self._release_cam()
                self._cap = self._open_cam()
                if self._cap is None:
                    return None
            ok, frame = self._cap.read()
            if not ok or frame is None:         # handle dropped -> reopen once
                self._release_cam()
                self._cap = self._open_cam()
                if self._cap is None:
                    return None
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def capture_jpeg(self):
        """Grab one frame and JPEG-encode it. Returns bytes, or None if no camera."""
        img = self.capture_pil()
        if img is None:
            return None
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()


# ----------------------------------------------------------------------------
# HTTP server: /ping (heartbeat), /frame (live image), /health
# ----------------------------------------------------------------------------
def _make_handler(shared):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass   # silence per-request console logging

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            shared.mark_contact()   # ANY request from the dashboard counts as presence
            if path == "/ping":
                self._send_json({"ok": True, "role": "worker"})
            elif path == "/health":
                self._send_json({"ok": True, "role": "worker",
                                 "since_contact_s": round(shared.seconds_since_contact(), 1)})
            elif path == "/frame":
                jpg = shared.capture_jpeg()
                if jpg is None:
                    self._send_json({"error": "camera unavailable"}, status=503)
                    return
                t, h = shared.latest_env
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("X-Temp", str(round(float(t), 1)))
                self.send_header("X-Hum", str(round(float(h), 1)))
                self.send_header("X-Age", str(int(config.chicken_age_days(shared.cfg))))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self._send_json({"error": "not found"}, status=404)

    return Handler


def start_http_server(shared, port):
    httpd = ThreadingHTTPServer(("0.0.0.0", int(port)), _make_handler(shared))
    t = threading.Thread(target=httpd.serve_forever, name="worker-http", daemon=True)
    t.start()
    return httpd


# ----------------------------------------------------------------------------
# Capture + commit (autonomous; also used by Telegram /capture)
# ----------------------------------------------------------------------------
def autonomous_cycle(cfg, cv_bundle, method, shared, alert_state):
    """Capture one frame (persistent handle), analyse, log + alert. Returns (row, coverage) or None."""
    img = shared.capture_pil()
    if img is None:
        print("[worker] no camera frame; skipping cycle")
        return None
    temp, hum = _read_env(cfg)
    shared.latest_env = (temp, hum)
    age = config.chicken_age_days(cfg)
    row, coverage, _imgs, _ex = engine.analyze_image(cfg, cv_bundle, img, method, temp, hum, age)
    synced = engine.commit_reading(cfg, row, coverage, source="worker", alert_state=alert_state)
    print(f"[worker] (auto) feed={row['feed_kg']}kg cover={coverage:.0f}% "
          f"birds={row['chicken_count']} THI={row['thi_c']} cloud={'ok' if synced else 'local-only'}")
    return row, coverage


# ----------------------------------------------------------------------------
# Two-way Telegram: /status, /current, /latest, /capture, /help
# ----------------------------------------------------------------------------
def _send(cfg, text):
    notify.send_telegram(cfg["telegram_bot_token"], cfg["telegram_chat_id"], text)


def _latest_from_db(cfg):
    """Most recent reading from the DB (logged by whichever side was active)."""
    rows = db.fetch_readings(cfg, limit=1)
    return rows[0] if rows else None


def _reply_latest(cfg):
    row = _latest_from_db(cfg)
    if row:
        _send(cfg, engine.status_msg(row, row.get("coverage_pct")))
    else:
        _send(cfg, "No readings yet — send /capture to grab one now.")


def _reply_capture(cfg, cv_bundle, method, shared, alert_state):
    _send(cfg, "\U0001f4f8 Capturing a fresh frame…")
    result = autonomous_cycle(cfg, cv_bundle, method, shared, alert_state)
    if result:
        row, coverage = result
        _send(cfg, engine.status_msg(row, coverage))
    else:
        _send(cfg, "⚠️ Couldn't capture — camera unavailable (busy/unplugged).")


def handle_telegram_commands(cfg, cv_bundle, method, offset, shared, alert_state):
    """Poll for commands and reply. Only the configured chat_id is honoured.
    Returns the new offset (last processed update_id + 1)."""
    if not config.is_telegram_configured(cfg):
        return offset
    token = cfg["telegram_bot_token"]
    owner = str(cfg["telegram_chat_id"])
    updates, ok = notify.get_updates(token, offset=offset)
    if not ok:
        return offset
    present_timeout = float(cfg.get("presence_timeout_sec", 25))
    for u in updates:
        offset = u["update_id"] + 1
        msg = u.get("message") or u.get("edited_message") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        text = (msg.get("text") or "").strip().lower()
        if chat != owner or not text.startswith("/"):
            continue   # ignore other chats / non-commands (security)
        cmd = text.split()[0].split("@")[0]   # "/current status" -> "/current"; strip @botname
        if cmd in ("/status", "/current"):
            # Read-only: NEVER capture/log here (that would violate stand-down while the
            # dashboard is present). Use /capture to force a fresh frame.
            row = _latest_from_db(cfg)
            if row:
                where = "dashboard is live" if shared.seconds_since_contact() < present_timeout else "worker logging"
                _send(cfg, engine.status_msg(row, row.get("coverage_pct")) + f"\n<i>({where})</i>")
            else:
                _send(cfg, "No readings yet — send /capture to grab one now.")
        elif cmd == "/latest":
            _reply_latest(cfg)
        elif cmd == "/capture":
            _reply_capture(cfg, cv_bundle, method, shared, alert_state)
        elif cmd in ("/help", "/start"):
            _send(cfg, "\U0001f414 <b>Broiler Monitor</b> commands:\n"
                       "/status – latest reading (live or worker-logged)\n"
                       "/latest – last saved reading\n"
                       "/capture – grab a fresh frame now")
        print(f"[worker] telegram cmd: {cmd}")
    return offset


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    cfg = config.load()
    method = int(cfg.get("feeder_method", 1))
    if not inference.feeder_method_available(method):
        print(f"[worker] feeder_method={method} model missing — falling back to Method 1.")
        method = 1
        cfg["feeder_method"] = 1
    cv_bundle = cv_feeder.CVConfigBundle.load(cv_feeder.config_path_for(method))
    interval = int(cfg.get("interval_sec", 600))
    presence_timeout = float(cfg.get("presence_timeout_sec", 25))
    port = int(cfg.get("worker_http_port", 8077))
    poll = max(1.0, min(5.0, float(interval)))   # loop cadence (presence + telegram check)

    # Start the HTTP server FIRST so /frame works immediately — it only needs the camera,
    # not the YOLO models. (Important on the Jetson: model warm-up below is ~30-90s, and we
    # don't want live frames blocked behind it.)
    shared = Shared(cfg)
    try:
        start_http_server(shared, port)
        print(f"[worker] HTTP frame/heartbeat server on :{port}  (/ping  /frame  /health)")
    except Exception as e:
        print(f"[worker] !! could not start HTTP server on :{port}: {e}")
        print("[worker] continuing as a standalone autonomous logger (no live streaming to a dashboard).")

    # Pre-warm the YOLO models so the first AUTONOMOUS capture isn't a slow cold-load.
    # On a Jetson Nano this one-time load is ~30-90s; the frame server above is already
    # serving while this runs. (The dashboard does its own model loading on the laptop.)
    try:
        print("[worker] loading models (one-time warm-up, can take ~30-90s on Jetson)…")
        inference.load_feeder_model(method)
        inference.load_chicken_model()
        print("[worker] models ready.")
    except Exception as e:
        print(f"[worker] model warm-up skipped ({e}); will load on first use.")

    # Pre-open the camera so the FIRST /frame is fast — opening it costs ~15-20s on a Nano,
    # and we don't want the dashboard's first live frame to stall on that.
    try:
        print("[worker] opening camera (one-time warm-up)…")
        print("[worker] camera ready." if shared.capture_pil() is not None
              else "[worker] camera not available yet — will open on demand.")
    except Exception as e:
        print(f"[worker] camera warm-up skipped ({e}).")

    print(f"[worker] started · method={method} · interval={interval}s · poll={poll:.0f}s · "
          f"presence_timeout={presence_timeout:.0f}s · "
          f"supabase={'on' if config.is_supabase_configured(cfg) else 'OFF'} · "
          f"telegram={'on' if config.is_telegram_configured(cfg) else 'OFF'}")
    print("[worker] dashboard ABSENT -> I capture + log; dashboard PRESENT -> I stand down (it logs). Waiting…")

    alert_state = {"last_low": 0.0, "last_heat": 0.0}
    last_capture_t = -1e9    # so the first autonomous tick captures immediately
    last_present = None      # log state changes only
    # Drain any Telegram backlog so we don't reply to stale commands on restart.
    tg_offset = None
    if config.is_telegram_configured(cfg):
        _u, _ok = notify.get_updates(cfg["telegram_bot_token"])
        if _ok and _u:
            tg_offset = _u[-1]["update_id"] + 1
    poll_count = 0
    reload_every = max(1, int(30 / poll))   # re-read monitor_config.json ~every 30s
    while True:
        try:
            # Hot-reload lightweight config (interval/thresholds/method) without restart;
            # keep YOLO/CV model loads OFF the hot path. Refresh the cached sensor reading
            # here too (so /frame carries a recent temp/hum without blocking every poll).
            if poll_count % reload_every == 0:
                cfg = config.load()
                shared.cfg = cfg
                new_method = int(cfg.get("feeder_method", 1))
                if not inference.feeder_method_available(new_method):
                    new_method = 1
                    cfg["feeder_method"] = 1
                if new_method != method:
                    method = new_method
                    cv_bundle = cv_feeder.CVConfigBundle.load(cv_feeder.config_path_for(method))
                interval = int(cfg.get("interval_sec", 600))
                presence_timeout = float(cfg.get("presence_timeout_sec", 25))
                shared.latest_env = _read_env(cfg)
            poll_count += 1

            # Answer Telegram commands in any state.
            tg_offset = handle_telegram_commands(cfg, cv_bundle, method, tg_offset, shared, alert_state)

            # Presence: if the dashboard is here, stand down (it does the logging).
            present = shared.seconds_since_contact() < presence_timeout
            if present != last_present:
                print("[worker] dashboard " + ("PRESENT — standing down (dashboard logs)."
                                               if present else "ABSENT — I am now logging."))
                last_present = present

            now = time.time()
            if (not present) and (now - last_capture_t) >= interval:
                result = None
                try:
                    result = autonomous_cycle(cfg, cv_bundle, method, shared, alert_state)
                except Exception as e:
                    print(f"[worker] capture/analyze failed: {e}")
                if result:
                    last_capture_t = now     # advance the timer only on a SUCCESSFUL capture
                else:
                    # transient capture/analyze failure: retry in ~max(poll,30)s, not a full interval
                    last_capture_t = now - max(0.0, interval - max(poll, 30.0))
        except Exception as e:
            print(f"[worker] cycle error: {e}")
        time.sleep(poll)


if __name__ == "__main__":
    main()
