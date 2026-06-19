#!/usr/bin/env bash
# Runs INSIDE the container on the JETSON. Split deployment: the Jetson runs ONLY
# the worker (engine) — camera capture, YOLO+CV, DB logging, Telegram alerts, and a
# tiny HTTP server (port 8077) that serves live frames + a presence heartbeat.
#
# The DASHBOARD runs on your LAPTOP (streamlit), not here. It reaches this worker by
# the Jetson's hostname over the LAN. Only the worker opens the camera.
set -u
cd /app

# Worker (engine): keep it alive — auto-restart if it ever crashes.
echo "[start] worker.py starting (HTTP frame/heartbeat server on :8077)…"
while true; do
    python3 worker.py || echo "[start] worker exited (code $?), restarting in 5s…"
    sleep 5
done
