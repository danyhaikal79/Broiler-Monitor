#!/usr/bin/env bash
# Run the Broiler Monitor dashboard in a container on the Jetson.
#
# - --runtime nvidia      : gives the container the Jetson GPU (for YOLO/TensorRT)
# - --device /dev/video0  : passes the USB webcam through to the container
# - -v "$PROJECT":/app     : mounts the whole project (models + config + code)
# - -p 8500:8500          : exposes the dashboard on the LAN
#
# Other PCs on the same network then open:  http://<jetson-ip>:8500
# (find the Jetson IP with:  hostname -I )
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
echo "[run] open    http://$(hostname -I | awk '{print $1}'):8500  from another PC on this network"

# Runs BOTH the worker (engine) + the dashboard (UI) via deploy/start_jetson.sh.
docker run --rm -it \
  --runtime nvidia \
  --network host \
  $DEV_FLAGS \
  -v "$PROJECT":/app \
  -w /app \
  "$IMAGE" \
  bash deploy/start_jetson.sh
