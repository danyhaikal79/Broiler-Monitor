#!/usr/bin/env bash
# Runs INSIDE the container. Starts BOTH processes that full_system3 needs:
#   1. worker.py  – the engine: camera capture, YOLO+CV, DB logging, Telegram alerts
#   2. streamlit  – the dashboard UI (a pure viewer/controller)
# They share /app, so the worker<->dashboard control/handoff files work.
# Only the worker opens the camera; the dashboard just reads what it publishes.
set -u
cd /app

# Worker (engine): keep it alive — auto-restart if it ever crashes.
( while true; do
    echo "[start] worker.py starting…"
    python3 worker.py || echo "[start] worker exited (code $?), restarting in 5s…"
    sleep 5
  done ) &

# Dashboard (UI) in the foreground. If it dies, the container exits and
# Docker/systemd restarts the whole thing.
echo "[start] dashboard on :8500 …"
exec python3 -m streamlit run new_dashboard/super_finalise_dashboard/app.py \
     --server.address 0.0.0.0 --server.port 8500
