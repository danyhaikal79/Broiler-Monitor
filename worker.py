"""
Background monitoring worker (runs on the Jetson, separate from the dashboard).

The worker is the ONLY place that does inference AND the ONLY thing that writes to
the database. The dashboard is a pure viewer/controller: it sets a flag and hands
over uploaded files; it never runs YOLO and never writes to the DB.

Each poll (~every few seconds) the worker:
  1. Processes any pending UPLOAD the dashboard handed over (image or video) —
     runs YOLO + CV, logs one reading, and publishes the annotated result back.
  2. In LIVE mode (control flag camera_enabled=True) captures a camera frame every
     `interval_sec`, logs it, publishes it, and fires a low-feed Telegram alert
     (with cooldown). In Upload mode the camera is PAUSED (but uploads still run).

Because the worker is its own process, it keeps logging after the dashboard/browser
is closed, as long as live mode was last selected. Only ONE process may own the
camera — don't run two workers.

Run:
  python3 worker.py
"""

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "new_dashboard"))

from PIL import Image  # noqa: E402
from core import config, db, notify, inference, cv_feeder, logic, video, sensor, control  # noqa: E402


def _read_env(cfg):
    """(temp, hum): from the DHT-22 if use_sensor, else config fallback."""
    if cfg.get("use_sensor"):
        reading, err = sensor.read_latest(cfg.get("serial_port", "/dev/ttyUSB0"),
                                          int(cfg.get("serial_baud", 115200)))
        if reading:
            return reading[0], reading[1]
        print(f"[worker] sensor read failed ({err}); using fallback")
    return float(cfg.get("temperature_fallback_c", 25.0)), float(cfg.get("humidity_fallback_pct", 70.0))


def _analyze_image(cfg, cv_bundle, img, method, temp, hum, age):
    """Core analysis of ONE PIL image -> (reading_row, coverage, images, extras).
    `images` = the 4 readout views; `extras` = extra display numbers. Shared by live
    capture and upload processing. Does NOT log or publish."""
    feeder = inference.run_feeder(img, method=method)
    chicken = inference.run_chicken(img)
    ftype = feeder.feeder_type or cfg.get("feeder_type", "pan7kg")
    cvm = cv_feeder.measure(img, cv_bundle, ftype,
                            yolo_polygon=feeder.feeder_polygon, yolo_bbox=feeder.feeder_bbox)
    fill = cvm.fill_ratio if cvm else 0.0
    _ov = cfg.get("chicken_count_override")
    count = _ov if _ov is not None else chicken.count
    pred = logic.compute(temperature_c=temp, humidity_pct=hum, age_days=age, chicken_count=count,
                         current_food_kg=cvm.current_food_kg if cvm else 0.0, feeder_type=ftype)
    coverage = (100.0 * pred.current_food_kg / pred.flock_daily_required_kg
                if pred.flock_daily_required_kg > 0 else 100.0)
    row = {
        "ts": db.now_iso(cfg.get("display_tz_offset_hours", 0)),
        "feeder_type": ftype, "feed_kg": round(pred.current_food_kg, 3),
        "fill_ratio": round(fill, 4), "chicken_count": int(count),
        "temperature_c": round(temp, 1), "humidity_pct": round(hum, 1),
        "thi_c": round(pred.thi_c, 2), "required_kg": round(pred.flock_daily_required_kg, 3),
        "feed_to_add_kg": round(pred.feed_to_add_kg, 3), "coverage_pct": round(coverage, 1),
    }
    images = {
        "original": img,
        "feeder": feeder.annotated_image,
        "feed_mask": (cvm.mask_image if cvm else None),
        "chickens": chicken.annotated_image,
    }
    extras = {
        "wet_bulb_c": round(pred.wet_bulb_c, 2),
        "thi_stress_factor": round(pred.thi_stress_factor, 3),
        "feeder_conf": round(feeder.feeder_type_conf, 3),
        "chicken_conf": round(chicken.avg_confidence, 3),
        "detected": feeder.feeder_type,   # None if no feeder detected
    }
    return row, coverage, images, extras


def _low_feed_msg(row, coverage, tag=""):
    head = "⚠️ <b>Feed low</b>" + (f" ({tag})" if tag else "")
    return (f"{head}\n"
            f"Feeder: {row['feeder_type']}\n"
            f"Current: {row['feed_kg']} kg ({coverage:.0f}% of today's need)\n"
            f"Add ~{row['feed_to_add_kg']} kg.\n"
            f"Birds: {row['chicken_count']} · THI: {row['thi_c']}°C")


def _heat_msg(row, threshold):
    return ("🔥 <b>HEAT EMERGENCY</b>\n"
            f"Temperature {row['temperature_c']}°C ≥ {threshold:.0f}°C — chickens are too hot "
            "(survival zone).\n"
            "Cool the house NOW: fans, water, ventilation.\n"
            f"THI {row['thi_c']}°C · RH {row['humidity_pct']}%")


def _status_msg(row, coverage, age_s=None):
    """Current-status reply (same style as the alerts)."""
    age = f"  ·  {int(age_s)}s ago" if age_s is not None else ""
    return ("📊 <b>Current status</b>" + age + "\n"
            f"Feeder: {row.get('feeder_type')}\n"
            f"Feed: {row.get('feed_kg')} kg ({float(coverage or 0):.0f}% of today's need)\n"
            f"Add: {row.get('feed_to_add_kg')} kg\n"
            f"Birds: {row.get('chicken_count')}\n"
            f"Temp {row.get('temperature_c')}°C · RH {row.get('humidity_pct')}% · "
            f"THI {row.get('thi_c')}°C")


def analyze_once(cfg, cv_bundle, method):
    """Capture one camera frame, analyze, log, and publish for the live view."""
    img = video.capture_frame(int(cfg["camera_index"]))
    if img is None:
        print("[worker] no camera frame; skipping cycle")
        return None
    ovr = control.read_control().get("env_override")
    if ovr:   # dashboard manual override (Live mode, no-sensor demo)
        temp = float(ovr.get("temp", cfg.get("temperature_fallback_c", 25.0)))
        hum = float(ovr.get("hum", cfg.get("humidity_fallback_pct", 70.0)))
        age = int(ovr.get("age", config.chicken_age_days(cfg)))
    else:
        temp, hum = _read_env(cfg)
        age = config.chicken_age_days(cfg)
    row, coverage, images, extras = _analyze_image(cfg, cv_bundle, img, method, temp, hum, age)
    synced = db.log_reading(cfg, row)
    control.write_latest(row, coverage, images=images, extras=extras)
    print(f"[worker] feed={row['feed_kg']}kg cover={coverage:.0f}% "
          f"birds={row['chicken_count']} THI={row['thi_c']} cloud={'ok' if synced else 'local-only'}")
    return row, coverage


# ----------------------------------------------------------------------------
# Two-way Telegram: respond to /status, /current, /latest, /capture, /help
# ----------------------------------------------------------------------------
def _send(cfg, text):
    notify.send_telegram(cfg["telegram_bot_token"], cfg["telegram_chat_id"], text)


def _reply_latest(cfg):
    latest = control.read_latest()
    if latest and latest.get("reading"):
        _send(cfg, _status_msg(latest["reading"], latest.get("coverage"), age_s=latest.get("age_s")))
    else:
        _send(cfg, "No saved reading yet — start live monitoring or send /capture.")


def _reply_capture(cfg, cv_bundle, method):
    _send(cfg, "📸 Capturing a fresh frame…")
    result = analyze_once(cfg, cv_bundle, method)   # one-off capture; logs + publishes
    if result:
        row, coverage = result
        _send(cfg, _status_msg(row, coverage))
    else:
        _send(cfg, "⚠️ Couldn't capture — camera unavailable (busy/unplugged).")


def handle_telegram_commands(cfg, cv_bundle, method, offset):
    """Poll for commands and reply. Only the configured chat_id is honored.
    Returns the new offset (last processed update_id + 1)."""
    if not config.is_telegram_configured(cfg):
        return offset
    token = cfg["telegram_bot_token"]
    owner = str(cfg["telegram_chat_id"])
    updates, ok = notify.get_updates(token, offset=offset)
    if not ok:
        return offset
    for u in updates:
        offset = u["update_id"] + 1
        msg = u.get("message") or u.get("edited_message") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        text = (msg.get("text") or "").strip().lower()
        if chat != owner or not text.startswith("/"):
            continue   # ignore other chats / non-commands (security)
        cmd = text.split()[0].split("@")[0]   # "/current status" -> "/current"; strip @botname
        if cmd in ("/status", "/current"):
            if control.read_control().get("camera_enabled"):
                latest = control.read_latest()
                if latest and latest.get("reading"):
                    _send(cfg, _status_msg(latest["reading"], latest.get("coverage"), age_s=latest.get("age_s")))
                else:
                    _reply_capture(cfg, cv_bundle, method)
            else:
                _send(cfg, "🔸 <b>Monitoring is paused</b> (Upload mode).\n"
                           "Send <b>/latest</b> for the last saved reading, or "
                           "<b>/capture</b> to grab a fresh frame now.")
        elif cmd == "/latest":
            _reply_latest(cfg)
        elif cmd == "/capture":
            _reply_capture(cfg, cv_bundle, method)
        elif cmd in ("/help", "/start"):
            _send(cfg, "🐔 <b>Broiler Monitor</b> commands:\n"
                       "/status – current data (live) or options (if paused)\n"
                       "/latest – last saved reading\n"
                       "/capture – grab a fresh frame now")
        print(f"[worker] telegram cmd: {cmd}")
    return offset


def process_upload(cfg, req):
    """Analyze + log an image/video the dashboard handed over, then publish the
    result (correlated by req id). For video, logs the peak-bird frame."""
    rid = req.get("id", "?")
    try:
        method = int(req.get("method", cfg.get("feeder_method", 1)))
        if method == 2 and not inference.feeder_method_available(2):
            method = 1
        cv_bundle = cv_feeder.CVConfigBundle.load(cv_feeder.config_path_for(method))
        temp = float(req.get("temp", cfg.get("temperature_fallback_c", 25.0)))
        hum = float(req.get("hum", cfg.get("humidity_fallback_pct", 70.0)))
        age = int(req.get("age", config.chicken_age_days(cfg)))
        data = req["data_bytes"]

        if req.get("kind") == "video":
            frames = video.sample_frames(data, max_frames=int(req.get("max_frames", 20)))
            if not frames:
                control.write_upload_result(rid, None, 0.0, error="could not read video frames")
                return
            best = None   # pick the peak-bird frame (best flock-size estimate)
            for _ts, fr in frames:
                r, cov, imgs, ex = _analyze_image(cfg, cv_bundle, fr, method, temp, hum, age)
                if best is None or r["chicken_count"] > best[0]["chicken_count"]:
                    best = (r, cov, imgs, ex)
            row, coverage, images, extras = best
        else:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            row, coverage, images, extras = _analyze_image(cfg, cv_bundle, img, method, temp, hum, age)

        db.log_reading(cfg, row)
        if coverage < float(cfg.get("low_feed_coverage_pct", 25)) and row["chicken_count"] > 0:
            if config.is_telegram_configured(cfg):
                notify.send_telegram(cfg["telegram_bot_token"], cfg["telegram_chat_id"],
                                     _low_feed_msg(row, coverage, "manual upload"))
            db.log_alert(cfg, "low_feed",
                         f"(upload) coverage {coverage:.0f}% · add {row['feed_to_add_kg']}kg", value=coverage)
        control.write_upload_result(rid, row, coverage, images=images, extras=extras)
        print(f"[worker] processed upload {rid[:8]} · feed={row['feed_kg']}kg cover={coverage:.0f}%")
    except Exception as e:
        print(f"[worker] upload {rid[:8]} failed: {e}")
        try:
            control.write_upload_result(rid, None, 0.0, error=str(e))
        except Exception:
            pass


def main():
    cfg = config.load()
    method = int(cfg.get("feeder_method", 1))
    if not inference.feeder_method_available(method):
        print(f"[worker] feeder_method={method} model missing — falling back to Method 1.")
        method = 1
        cfg["feeder_method"] = 1
    cv_bundle = cv_feeder.CVConfigBundle.load(cv_feeder.config_path_for(method))
    interval = int(cfg.get("interval_sec", 600))
    low_thr = float(cfg.get("low_feed_coverage_pct", 25))
    cooldown = float(cfg.get("alert_cooldown_min", 60)) * 60
    high_temp = float(cfg.get("high_temp_alert_c", 38))
    poll = max(1.0, min(5.0, float(interval)))   # control-flag + upload-inbox check cadence

    print(f"[worker] started · method={method} · interval={interval}s · poll={poll:.0f}s · "
          f"low-feed<{low_thr}% · supabase={'on' if config.is_supabase_configured(cfg) else 'OFF'} · "
          f"telegram={'on' if config.is_telegram_configured(cfg) else 'OFF'}")
    print("[worker] live mode = capture; upload mode = camera paused (uploads still processed). Waiting…")

    last_alert_t = 0.0
    last_heat_alert_t = 0.0
    last_capture_t = -1e9    # so the first enabled tick captures immediately
    last_enabled = None      # to log state changes only
    last_upload_id = None    # so each handed-over upload is processed once
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
            # keep YOLO/CV model loads OFF the hot path.
            if poll_count % reload_every == 0:
                cfg = config.load()
                new_method = int(cfg.get("feeder_method", 1))
                if not inference.feeder_method_available(new_method):
                    new_method = 1
                    cfg["feeder_method"] = 1
                if new_method != method:
                    method = new_method
                    cv_bundle = cv_feeder.CVConfigBundle.load(cv_feeder.config_path_for(method))
                interval = int(cfg.get("interval_sec", 600))
                low_thr = float(cfg.get("low_feed_coverage_pct", 25))
                cooldown = float(cfg.get("alert_cooldown_min", 60)) * 60
                high_temp = float(cfg.get("high_temp_alert_c", 38))
            poll_count += 1

            # 1) Pending upload handed over by the dashboard (runs in ANY mode).
            req = control.read_upload_request()
            if req and req.get("id") and req["id"] != last_upload_id:
                last_upload_id = req["id"]
                process_upload(cfg, req)
                control.consume_upload_request()   # so a worker restart won't re-run it

            # 1b) Answer Telegram commands (/status, /latest, /capture) in any mode.
            tg_offset = handle_telegram_commands(cfg, cv_bundle, method, tg_offset)

            # 2) Live capture when enabled and it's time.
            enabled = control.read_control().get("camera_enabled", False)
            if enabled != last_enabled:
                print(f"[worker] camera {'ENABLED (live mode)' if enabled else 'PAUSED (upload mode)'}")
                last_enabled = enabled
            now = time.time()
            if enabled and (now - last_capture_t) >= interval:
                result = None
                try:
                    result = analyze_once(cfg, cv_bundle, method)
                except Exception as e:
                    print(f"[worker] capture/analyze failed: {e}")
                if result:
                    last_capture_t = now     # advance the timer only on a SUCCESSFUL capture
                    row, coverage = result
                    if coverage < low_thr and row["chicken_count"] > 0 and now - last_alert_t >= cooldown:
                        if config.is_telegram_configured(cfg):
                            ok, info = notify.send_telegram(
                                cfg["telegram_bot_token"], cfg["telegram_chat_id"], _low_feed_msg(row, coverage))
                            print(f"[worker] ALERT sent: {ok} ({info})")
                        db.log_alert(cfg, "low_feed",
                                     f"coverage {coverage:.0f}% · add {row['feed_to_add_kg']}kg", value=coverage)
                        last_alert_t = now   # arm cooldown on LOG (no DB spam), Telegram configured or not
                    if row["temperature_c"] >= high_temp and now - last_heat_alert_t >= cooldown:
                        if config.is_telegram_configured(cfg):
                            ok, info = notify.send_telegram(
                                cfg["telegram_bot_token"], cfg["telegram_chat_id"], _heat_msg(row, high_temp))
                            print(f"[worker] HEAT ALERT sent: {ok} ({info})")
                        db.log_alert(cfg, "high_temp",
                                     f"temp {row['temperature_c']}°C ≥ {high_temp:.0f}°C — heat emergency",
                                     value=row["temperature_c"])
                        last_heat_alert_t = now
                else:
                    # transient capture/analyze failure: retry in ~max(poll,30)s, not a full interval
                    last_capture_t = now - max(0.0, interval - max(poll, 30.0))
        except Exception as e:
            print(f"[worker] cycle error: {e}")
        time.sleep(poll)


if __name__ == "__main__":
    main()
