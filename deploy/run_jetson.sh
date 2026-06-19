#!/usr/bin/env bash
# Run the Broiler Monitor WORKER in a container on the Jetson (split deployment).
# The dashboard runs separately on your laptop and connects to this worker over the LAN.
#
# - --runtime nvidia      : gives the container the Jetson GPU (for YOLO)
# - --device /dev/video0  : passes the USB webcam through to the container
# - --device /dev/ttyUSB0 : passes the ESP32/DHT-22 serial through
# - -v "$PROJECT":/app     : mounts the whole project (models + config + code)
# - --network host        : the worker's HTTP server (:8077) is reachable on the LAN
#
# From the LAPTOP dashboard, set worker_host = this Jetson's hostname (e.g.
# jetson00-desktop.local). Sanity-check the worker from any PC on the network:
#   http://<jetson-hostname>:8077/health
#
# Usage:
#   bash deploy/run_jetson.sh                 # uses IMAGE below
#   IMAGE=dustynv/ultralytics:r32.7.1 bash deploy/run_jetson.sh   # run base image directly + pip install at start

set -e

# The image you built (see deploy/README_JETSON.md). Override via env var.
IMAGE="${IMAGE:-broiler-monitor:jetson}"

# Absolute path to the project root (the folder containing feeder_train/, new_dashboard/, etc.)
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"

# USB webcam (change to /dev/video1 etc. if it enumerates differently)
CAM="${CAM:-/dev/video0}"
# ESP32 / DHT-22 USB serial (on Linux the ESP32 shows up as /dev/ttyUSB0, sometimes ttyACM0)
SERIAL="${SERIAL:-/dev/ttyUSB0}"

# Build device flags only for devices that actually exist (so a missing camera or
# unplugged ESP32 doesn't break the run).
DEV_FLAGS=""
[ -e "$CAM" ]    && DEV_FLAGS="$DEV_FLAGS --device $CAM"    || echo "[warn] $CAM not found (USB camera)"
[ -e "$SERIAL" ] && DEV_FLAGS="$DEV_FLAGS --device $SERIAL" || echo "[warn] $SERIAL not found (ESP32/DHT-22)"

echo "[run] image   = $IMAGE"
echo "[run] project = $PROJECT"
echo "[run] devices =$DEV_FLAGS"
echo "[run] worker HTTP on :8077 — check from another PC:  http://$(hostname):8077/health"

# Runs the worker (engine + frame/heartbeat server) via deploy/start_jetson.sh.
docker run --rm -it \
  --runtime nvidia \
  --network host \
  $DEV_FLAGS \
  -v "$PROJECT":/app \
  -w /app \
  "$IMAGE" \
  bash deploy/start_jetson.sh
