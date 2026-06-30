"""
Shared analysis + commit pipeline used by BOTH the worker (Jetson) and the
dashboard (laptop), so the two machines produce IDENTICAL readings, alerts and
messages — there is only one copy of the logic.

  analyze_image  : YOLO (feeder + chickens) + classical-CV feed level + the
                   feed/heat model on ONE PIL image -> (row, coverage, images, extras)
  analyze_upload : same, for an uploaded image or video (video -> peak-bird frame)
  commit_reading : log the reading to the DB and fire the low-feed + heat Telegram
                   alerts (cooldown is held in the caller's alert_state dict)
  *_msg          : alert / status message text

This module does NO camera or sensor I/O — callers pass temp / hum / age in.
Kept Python-3.8 safe (runs on the Jetson) via `from __future__ import annotations`.
"""

from __future__ import annotations

import io
import time

from PIL import Image

from . import config, db, notify, inference, cv_feeder, logic, video  # noqa: F401


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def analyze_image(cfg, cv_bundle, img, method, temp, hum, age):
    """Analyse ONE PIL image -> (reading_row, coverage, images, extras).
    `images` = the 4 readout views (PIL); `extras` = extra display numbers."""
    feeder = inference.run_feeder(img, method=method)
    chicken = inference.run_chicken(img, method=method)
    detected = feeder.feeder_type                          # None if no feeder is in view
    ftype = detected or cfg.get("feeder_type", "pan7kg")   # a valid type for the CV / feed math
    if detected is None:
        # No feeder detected -> the feed reading is meaningless. Report 0 kg instead of
        # measuring a fallback ROI of bedding (which produced a phantom non-zero weight).
        cvm, fill, food_kg = None, 0.0, 0.0
    else:
        cvm = cv_feeder.measure(img, cv_bundle, ftype,
                                yolo_polygon=feeder.feeder_polygon, yolo_bbox=feeder.feeder_bbox)
        fill = cvm.fill_ratio if cvm else 0.0
        food_kg = cvm.current_food_kg if cvm else 0.0
    _ov = cfg.get("chicken_count_override")
    count = _ov if _ov is not None else chicken.count
    pred = logic.compute(temperature_c=temp, humidity_pct=hum, age_days=age, chicken_count=count,
                         current_food_kg=food_kg, feeder_type=ftype)
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


def analyze_upload(cfg, cv_bundle, data, kind, method, temp, hum, age, max_frames=20):
    """Analyse an uploaded image or video (raw bytes). For a video, samples frames
    and returns the PEAK-bird frame (best flock-size estimate). Raises on bad input."""
    if kind == "video":
        frames = video.sample_frames(data, max_frames=int(max_frames))
        if not frames:
            raise ValueError("could not read video frames")
        best = None
        for _ts, fr in frames:
            r, cov, imgs, ex = analyze_image(cfg, cv_bundle, fr, method, temp, hum, age)
            if best is None or r["chicken_count"] > best[0]["chicken_count"]:
                best = (r, cov, imgs, ex)
        return best
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return analyze_image(cfg, cv_bundle, img, method, temp, hum, age)


# ----------------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------------
def low_feed_msg(row, coverage, tag=""):
    head = "⚠️ <b>Feed low</b>" + (f" ({tag})" if tag else "")
    return (f"{head}\n"
            f"Feeder: {row['feeder_type']}\n"
            f"Current: {row['feed_kg']} kg ({coverage:.0f}% of today's need)\n"
            f"Add ~{row['feed_to_add_kg']} kg.\n"
            f"Birds: {row['chicken_count']} · THI: {row['thi_c']}°C")


def heat_msg(row, threshold):
    return ("\U0001f525 <b>HEAT EMERGENCY</b>\n"
            f"Temperature {row['temperature_c']}°C ≥ {threshold:.0f}°C — chickens are too hot "
            "(survival zone).\n"
            "Cool the house NOW: fans, water, ventilation.\n"
            f"THI {row['thi_c']}°C · RH {row['humidity_pct']}%")


def status_msg(row, coverage, age_s=None):
    """Current-status reply (same style as the alerts)."""
    age = f"  ·  {int(age_s)}s ago" if age_s is not None else ""
    return ("\U0001f4ca <b>Current status</b>" + age + "\n"
            f"Feeder: {row.get('feeder_type')}\n"
            f"Feed: {row.get('feed_kg')} kg ({float(coverage or 0):.0f}% of today's need)\n"
            f"Add: {row.get('feed_to_add_kg')} kg\n"
            f"Birds: {row.get('chicken_count')}\n"
            f"Temp {row.get('temperature_c')}°C · RH {row.get('humidity_pct')}% · "
            f"THI {row.get('thi_c')}°C")


# ----------------------------------------------------------------------------
# Commit: log + alerts (cooldown shared across worker + dashboard via the DB)
# ----------------------------------------------------------------------------
def _recent_alert(cfg, kind, cooldown_s) -> bool:
    """True if an alert of `kind` was logged within the last `cooldown_s` seconds.

    DB-backed so the cooldown is SHARED across the worker (Jetson) and the dashboard
    (laptop) — they write the same Supabase alerts table, so neither re-fires an alert
    the other just sent (e.g. on a present<->absent handoff). Best-effort: any error
    (offline, parse) -> False, and the caller's in-process gate still applies."""
    try:
        from datetime import datetime, timedelta, timezone
        off = float(cfg.get("display_tz_offset_hours", 0))
        now_local = (datetime.now(timezone.utc) + timedelta(hours=off)).replace(tzinfo=None)
        for a in db.fetch_alerts(cfg, limit=25):   # newest first
            if a.get("kind") != kind:
                continue
            t = datetime.fromisoformat(str(a.get("ts"))[:19])   # stored as naive local wall-clock
            return (now_local - t).total_seconds() < cooldown_s
        return False
    except Exception:
        return False


def commit_reading(cfg, row, coverage, *, source="worker", alert_state=None, feeder_detected=True):
    """Log one reading to the DB, then fire the low-feed + heat-emergency alerts.

    Cooldown is enforced two ways: a cheap in-process gate (`alert_state`, owned by the
    caller — worker keeps one in memory, dashboard one in st.session_state) so we don't
    re-query every reading, AND an authoritative DB check (`_recent_alert`) so the worker
    and dashboard share one cooldown and never double-alert across a handoff. `source`
    tags origin ("worker" | "live" | "upload"). Returns True if the reading reached the cloud."""
    if alert_state is None:
        alert_state = {}
    synced = db.log_reading(cfg, row)

    now = time.time()
    low_thr = float(cfg.get("low_feed_coverage_pct", 25))
    high_temp = float(cfg.get("high_temp_alert_c", 38))
    cooldown = float(cfg.get("alert_cooldown_min", 60)) * 60
    tg = config.is_telegram_configured(cfg)
    tag = "" if source == "worker" else source

    # low-feed alert only when a feeder was actually detected: no feeder in view means the
    # feed is "unavailable", not "empty" — don't cry wolf when the cup is just out of frame.
    if coverage < low_thr and int(row.get("chicken_count", 0)) > 0 and feeder_detected \
            and now - alert_state.get("last_low", 0.0) >= cooldown:
        if not _recent_alert(cfg, "low_feed", cooldown):   # not already sent by either side
            if tg:
                notify.send_telegram(cfg["telegram_bot_token"], cfg["telegram_chat_id"],
                                     low_feed_msg(row, coverage, tag))
            db.log_alert(cfg, "low_feed",
                         f"({source}) coverage {coverage:.0f}% · add {row['feed_to_add_kg']}kg", value=coverage)
        alert_state["last_low"] = now   # arm in-process gate so we don't re-query every reading

    if float(row.get("temperature_c", 0)) >= high_temp \
            and now - alert_state.get("last_heat", 0.0) >= cooldown:
        if not _recent_alert(cfg, "high_temp", cooldown):
            if tg:
                notify.send_telegram(cfg["telegram_bot_token"], cfg["telegram_chat_id"],
                                     heat_msg(row, high_temp))
            db.log_alert(cfg, "high_temp",
                         f"({source}) temp {row['temperature_c']}°C ≥ {high_temp:.0f}°C — heat emergency",
                         value=row["temperature_c"])
        alert_state["last_heat"] = now

    return synced
