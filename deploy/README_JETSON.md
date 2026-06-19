# Deploying Broiler Monitor — split: Jetson worker + laptop dashboard

**Topology (current design):**

- **Jetson** runs **`worker.py` only** — it owns the USB camera + DHT-22, logs to the
  database 24/7 when you're away, and serves a tiny HTTP API on **port 8077**
  (`/frame`, `/ping`, `/health`).
- **Laptop** runs the **dashboard** (Streamlit). It does its own YOLO + CV inference.
  In **Live** mode it pulls frames from the Jetson over the LAN; in **Upload** mode it
  analyzes files locally. While the dashboard is open the worker stands down and the
  laptop does the logging.
- Both are on the **same Wi-Fi/hotspot**. The laptop addresses the Jetson by its
  **mDNS hostname** (e.g. `jetson00-desktop.local`), so a changing IP never matters.

---

## ⚠️ Read this first — the Jetson version determines the base image

The **original** Jetson Nano (2019) ships JetPack 4.x → Ubuntu 18.04 → Python 3.6,
too old for modern Ultralytics. Docker solves this with a Python-3.8 + PyTorch-CUDA
container. The base image tag depends on your exact L4T version, so we find it first.

### Step 0 — find your JetPack / L4T version (run ON the Jetson)

```bash
cat /etc/nv_tegra_release
# Example:  # R32 (release), REVISION: 7.1, ...  -> L4T r32.7.1  (JetPack 4.6.1)
```

The rest uses `r32.7.1` as the example — replace it with yours.

---

## Step 1 — Docker + NVIDIA runtime (usually already on JetPack)

```bash
docker --version
sudo docker info | grep -i runtime        # should list 'nvidia'
sudo usermod -aG docker $USER && newgrp docker   # run docker without sudo
```

## Step 2 — get an Ultralytics-on-Jetson base image

```bash
docker pull dustynv/ultralytics:r32.7.1
```

## Step 3 — build the thin image (from the project root `fyp3/`)

```bash
docker build -f deploy/Dockerfile.jetson \
    --build-arg BASE_IMAGE=dustynv/ultralytics:r32.7.1 \
    -t broiler-monitor:jetson .
```

The project is **mounted at runtime**, not copied — your models + config stay live.

---

## Step 4 — run the WORKER on the Jetson

Put `monitor_config.json` next to the project (Supabase + Telegram creds, `camera_index`,
`serial_port`, etc.). Then:

```bash
bash deploy/run_jetson.sh
```

This:
- gives the container the GPU (`--runtime nvidia`) and `--network host`
- passes the USB webcam (`CAM=/dev/video0`) and the ESP32/DHT-22 serial (`SERIAL=/dev/ttyUSB0`)
- mounts the project and runs **`worker.py`** (no Streamlit on the Jetson)

**Check it's up** — from the Jetson or any PC on the network, open:

```
http://jetson00-desktop.local:8077/health      # -> {"ok": true, "role": "worker", ...}
```

(Use your Jetson's hostname; `hostname` on the Jetson prints it.)

### Auto-start on boot (optional)

```bash
sudo cp deploy/broiler-monitor.service /etc/systemd/system/   # EDIT the project path + devices inside first
sudo systemctl daemon-reload
sudo systemctl enable --now broiler-monitor
journalctl -u broiler-monitor -f                              # live logs
```

---

## Step 5 — run the DASHBOARD on your laptop

On the laptop, in the project, make sure `monitor_config.json` has:

```json
"worker_host": "jetson00-desktop.local",   // your Jetson's hostname
"worker_http_port": 8077
```

(plus the same Supabase creds, so History/Alerts read the cloud DB). Then:

```bash
python -m streamlit run new_dashboard/super_finalise_dashboard/app.py
```

Open `http://localhost:8501`, log in (`admin/admin123` or `staff/staff123`), and on the
**Monitor** page choose **Upload file** or **Live camera**. The sidebar shows
**● WORKER ONLINE** when it can reach the Jetson.

> **All-in-one test (no Jetson):** set `worker_host` to `127.0.0.1`, run `worker.py`
> and the dashboard on the same machine. The laptop's own webcam serves `/frame`.

---

## Step 6 — make 4 GB survive (worker is much lighter than before)

Running **only** the worker (no Streamlit) on the Jetson frees a lot of RAM, but two
YOLO models are still heavy on a 4 GB Nano. Recommended:

```bash
# 4 GB swap (one-time)
sudo fallocate -l 4G /var/swapfile && sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile && sudo swapon /var/swapfile
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# headless (no desktop GUI) to free RAM
sudo systemctl set-default multi-user.target   # revert with graphical.target

# max clocks
sudo nvpmodel -m 0 && sudo jetson_clocks
```

The Nano's first model load is slow (~30–90 s) and per-frame inference is seconds — fine
for **autonomous interval logging**, but for a responsive **Live** view keep the laptop
open (it does the inference; the Jetson just sends frames).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard shows **WORKER OFFLINE** | Is `worker.py` running on the Jetson? Same Wi-Fi? Open `http://<jetson-host>:8077/health` in a browser. Check `worker_host` in the laptop's config. |
| `jetson00-desktop.local` won't resolve | mDNS issue. Confirm `avahi-daemon` runs on the Jetson; on Windows, Bonjour/mDNS must be available. As a quick test, try the Jetson's current IP. |
| Live view "camera unavailable (503)" | Camera busy/unplugged on the Jetson. Check `ls /dev/video*`; set `CAM=/dev/videoN` in `run_jetson.sh`. |
| `could not select device driver "nvidia"` | NVIDIA runtime not default. Add `"default-runtime": "nvidia"` to `/etc/docker/daemon.json`, `sudo systemctl restart docker`. |
| OOM / killed on the Jetson | Enable swap + headless (Step 6). The worker alone is much lighter than worker+dashboard was. |
| Worker logs even when dashboard is open | The dashboard's heartbeat isn't reaching the worker — check the browser can reach `:8077` (same network, no firewall blocking the port). |
