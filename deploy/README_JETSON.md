# Deploying Broiler Monitor on the Jetson Nano (Docker)

Goal: run the dashboard **on the Jetson** (with the USB webcam attached) so any
PC on the same network can open it at `http://<jetson-ip>:8500`.

---

## ⚠️ Read this first — the version determines everything

The **original** Jetson Nano (2019) ships JetPack 4.x → Ubuntu 18.04 → Python 3.6,
which is too old for modern Ultralytics. Docker solves this by running a container
with Python 3.8 + PyTorch-CUDA inside. But the **base image tag depends on your
exact L4T version**, so we find it first.

### Step 0 — find your JetPack / L4T version (run ON the Jetson)

```bash
cat /etc/nv_tegra_release
# Example output:  # R32 (release), REVISION: 7.1, ...   -> L4T r32.7.1  (JetPack 4.6.1)

# or:
sudo apt-cache show nvidia-jetpack 2>/dev/null | grep -m1 Version
```

**Tell me the L4T version** (e.g. `r32.7.1`) and I'll lock in the exact image tag.
The rest of this guide uses `r32.7.1` as the example — replace it with yours.

---

## Step 1 — install Docker + NVIDIA runtime (usually already on JetPack)

JetPack normally ships Docker + the NVIDIA container runtime. Verify:

```bash
docker --version
sudo docker info | grep -i runtime        # should list 'nvidia'
```

Add yourself to the docker group so you don't need sudo each time:

```bash
sudo usermod -aG docker $USER && newgrp docker
```

---

## Step 2 — get an Ultralytics-on-Jetson base image

The cleanest source is **dusty-nv's prebuilt containers**. For your L4T version,
pull the matching ultralytics image, e.g.:

```bash
docker pull dustynv/ultralytics:r32.7.1
```

> If no prebuilt tag exists for your exact L4T, two fallbacks:
> 1. Use `dusty-nv/jetson-containers` to build it (clone an older commit that still
>    supports r32 if you're on JetPack 4). Building on a 4GB Nano is slow — enable swap first (Step 5).
> 2. Use the closest `l4t-pytorch` / `l4t-ml` base and `pip install ultralytics` inside.
>
> Send me your version and I'll give the precise command.

---

## Step 3 — add Streamlit (build the thin image)

From the **project root** (`fyp3/`), build our image on top of the base:

```bash
docker build -f deploy/Dockerfile.jetson \
    --build-arg BASE_IMAGE=dustynv/ultralytics:r32.7.1 \
    -t broiler-monitor:jetson .
```

This just layers Streamlit on top — the project itself is **mounted at runtime**, not copied.

---

## Step 4 — run it

```bash
bash deploy/run_jetson.sh
```

This:
- gives the container the GPU (`--runtime nvidia`)
- passes the USB webcam (`--device /dev/video0` — change `CAM=/dev/video1` if needed)
- mounts the project so models + `cv_config.json` are live
- serves on port 8500 over the LAN

Then on another PC: open `http://<jetson-ip>:8500` (get the IP with `hostname -I`),
log in (`admin/admin123` or `staff/staff123`), pick **Live camera → Capture now**.

---

## Step 5 — make 4GB survive (important)

Two YOLO models + Streamlit is tight on 4 GB. Before running:

```bash
# Add 4GB swap (one-time)
sudo fallocate -l 4G /var/swapfile && sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile && sudo swapon /var/swapfile
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# Run headless (no desktop GUI) to free RAM
sudo systemctl set-default multi-user.target   # boots to console; revert with graphical.target

# Max performance / clocks
sudo nvpmodel -m 0 && sudo jetson_clocks
```

Expect **~1–3 s per analysis** with the "Capture now" button. Avoid auto-capture
on the Nano (re-runs both models on a timer → heat + RAM pressure). A **2GB Nano
will likely run out of memory** — if that's what you have, tell me and we'll
load the two models one-at-a-time or go the TensorRT-only route.

---

## Step 6 (optional, later) — TensorRT speed-up

No extra hardware — uses the Nano's built-in GPU. Build the engines **on the Nano**:

```bash
# inside the container, on the Jetson:
yolo export model=feeder_train/runs/seg_compare/yolov8n/weights/best.pt format=engine half=True imgsz=640
yolo export model=chicken_train/runs/chicken_compare/yolo11n/weights/best.pt format=engine half=True imgsz=640
```

Then point `inference.py` at the `.engine` files instead of `.pt`. Ask me and I'll
wire that switch in. Do this **after** the PyTorch path works.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "nvidia"` | NVIDIA runtime not set as default. `sudo nano /etc/docker/daemon.json` → add `"default-runtime": "nvidia"`, then `sudo systemctl restart docker`. |
| Camera not found in container | Check `ls /dev/video*` on the host; pass the right one via `CAM=/dev/videoN`. |
| OOM / killed | Enable swap (Step 5), run headless, close the desktop. Consider loading one model at a time. |
| Streamlit not reachable from other PC | Confirm `--server.address 0.0.0.0` (run script does this) and that you used the Jetson's LAN IP, not localhost. `--network host` is already set. |
| pip can't find streamlit version | base Python may be old; tell me the version and I'll pin a compatible Streamlit. |
