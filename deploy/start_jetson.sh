#!/usr/bin/env bash
# Runs INSIDE the container on the JETSON. Split deployment: the Jetson runs ONLY
# the worker (engine) — camera capture, YOLO+CV, DB logging, Telegram alerts, and a
# tiny HTTP server (port 8077) that serves live frames + a presence heartbeat.
#
# The DASHBOARD runs on your LAPTOP (streamlit), not here. It reaches this worker by
# the Jetson's hostname over the LAN. Only the worker opens the camera.
set -u
cd /app

# Stop cleanly on Ctrl+C (SIGINT) or `docker stop` (SIGTERM): set a flag and break the
# restart loop, so the script exits and the container shuts down instead of relaunching.
stop=0
trap 'stop=1; echo "[start] stop signal received — shutting down."' INT TERM

# Worker (engine): auto-restart if it CRASHES, but exit on Ctrl+C.
echo "[start] worker.py starting (HTTP frame/heartbeat server on :8077). Press Ctrl+C to stop."
while [ "$stop" -eq 0 ]; do
    python3 worker.py
    code=$?
    [ "$stop" -eq 1 ] && break          # Ctrl+C during the worker -> exit, don't restart
    echo "[start] worker exited (code $code), restarting in 5s… (Ctrl+C to stop)"
    sleep 5
done
echo "[start] stopped."
