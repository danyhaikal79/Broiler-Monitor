# Broiler Feed Monitoring — Full System (deployment)

Self-contained deployment package: YOLO feeder segmentation + classical-CV feed
level + YOLO chicken counting + Ross 308/THI feed prediction, served as a
role-gated Streamlit dashboard. Designed to run on a Jetson Nano via Docker and
be accessed from any device on the same network.

## What's in here

```
full_system/
├── new_dashboard/
│   ├── core/                       # engine: inference, CV, logic, theme, auth, sensor, video
│   │   │                           #   + finalise_view (monitor) + calibrate_view (admin)
│   │   └── cv_config.json          # calibrated per-feeder CV config (combined HSV+texture)
│   └── super_finalise_dashboard/app.py   # THE app: login + role-gated monitor/calibration
├── feeder_train/runs/seg_compare/yolov8n/weights/best.pt    # feeder segmentation model
├── chicken_train/runs/chicken_compare/yolo11n/weights/best.pt  # chicken detection model
├── deploy/                         # Jetson Docker deployment
│   ├── Dockerfile.jetson
│   ├── run_jetson.sh
│   └── README_JETSON.md            # full step-by-step Jetson guide
└── hardware/dht22_esp32/           # ESP32 firmware + wiring for the DHT-22 sensor
```

One entry point — `super_finalise_dashboard/app.py` — handles everything via login:
**admin** sees monitor + calibration, **staff** sees monitor only. (The monitor and
calibration screens live in `core/finalise_view.py` and `core/calibrate_view.py`.)

Models are kept (≈6 MB each, fine for git). Datasets are intentionally excluded.

## Run on a PC (dev/test)

```bash
pip install ultralytics streamlit pyserial streamlit-autorefresh pandas opencv-python
python -m streamlit run new_dashboard/super_finalise_dashboard/app.py --server.address 0.0.0.0 --server.port 8500
```
Login: `admin / admin123` (monitor + calibration) · `staff / staff123` (monitor only).

## Deploy on Jetson Nano (Docker)

See **`deploy/README_JETSON.md`** for the full guide. Short version, on the Jetson:

```bash
# 1. build the image (adds Streamlit on top of Ultralytics' JetPack-4 base)
docker build -f deploy/Dockerfile.jetson -t broiler-monitor:jetson .

# 2. run (passes USB camera + ESP32 serial + GPU + LAN port)
bash deploy/run_jetson.sh
```
Then open `http://<jetson-ip>:8500` from any PC on the same network.

## Hardware

- USB webcam → live frame capture
- ESP32 + DHT-22 over USB serial → live temperature/humidity (see `hardware/dht22_esp32/`)
