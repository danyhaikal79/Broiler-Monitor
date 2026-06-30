"""
Per-cycle COMPUTE benchmark — the same three stages the system runs every cycle:
    1. feeder segmentation   (inference.run_feeder)
    2. chicken detection      (inference.run_chicken)
    3. classical-CV feed read (cv_feeder.measure)

HARDWARE-AGNOSTIC — run it on BOTH machines on the SAME images so the only
difference is the hardware:

    laptop :  python  bench_compute.py  <image_folder> [method]
    Jetson :  python3 bench_compute.py  <image_folder> [method]

(optional [method]: 1=whole-feeder, 2=open-area; pass the SAME on both machines)

It loads exactly the deployed models (and auto-uses a TensorRT .engine if one
has been exported next to the .pt, just like the worker), warms up, then times
each stage over N cycles and prints the mean / min / max. Use feeder images so
the feed stage actually runs. With no folder it grabs one frame from the camera.
"""

import glob
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "new_dashboard"))

from PIL import Image  # noqa: E402
from core import config, inference, cv_feeder  # noqa: E402

WARMUP = 5      # discard: first inferences pay CUDA/TensorRT/cuDNN one-time cost
TIMED = 20      # timed cycles (matches the laptop measurement)


def load_images(arg):
    p = Path(arg)
    if p.is_dir():
        files = sorted(f for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                       for f in glob.glob(str(p / ext)))
    elif p.is_file():
        files = [str(p)]
    else:
        files = []
    return [Image.open(f).convert("RGB") for f in files]


def grab_one_from_camera(cfg):
    import cv2
    import platform
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(cfg.get("camera_index", 0)), backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame = None
    for _ in range(6):
        ok, frame = cap.read()
    cap.release()
    if frame is None:
        return []
    return [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]


def main():
    cfg = config.load()
    method = int(sys.argv[2]) if len(sys.argv) > 2 else int(cfg.get("feeder_method", 1))
    if not inference.feeder_method_available(method):
        print(f"method {method} weights missing -> falling back to Method 1")
        method = 1
    cv_bundle = cv_feeder.CVConfigBundle.load(cv_feeder.config_path_for(method), method)

    imgs = load_images(sys.argv[1]) if len(sys.argv) > 1 else []
    from_camera = False
    if not imgs:
        print("no image folder given (or empty) -> grabbing one frame from the camera")
        imgs = grab_one_from_camera(cfg)
        from_camera = True
    if not imgs:
        print("ERROR: no images and no camera frame; pass an image folder")
        sys.exit(1)
    # When grabbed from the camera, save it LOSSLESS (PNG) so the exact same frame can be
    # copied to another machine and benchmarked there for a truly identical comparison.
    if from_camera:
        try:
            imgs[0].save("bench_frame.png")
            print(f"saved captured frame -> {Path('bench_frame.png').resolve()}")
            print("   (copy this file to the other machine and run:  bench_compute.py bench_frame.png 3)")
        except Exception as e:
            print(f"(could not save frame: {e})")

    # device + which weights actually loaded (.pt vs TensorRT .engine)
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "CPU"
    except Exception:
        cuda, dev = False, "unknown"
    fw = inference._prefer_engine(inference.FEEDER_WEIGHTS_BY_METHOD.get(method, inference.FEEDER_WEIGHTS))
    cw = inference._prefer_engine(inference.CHICKEN_WEIGHTS_DEMO if method == 3 else inference.CHICKEN_WEIGHTS)
    print(f"device : {'CUDA' if cuda else 'CPU'} ({dev})", flush=True)
    print(f"method : {method}   feeder='{fw.name}'   chicken='{cw.name}'", flush=True)
    print(f"images : {len(imgs)}   warmup={WARMUP}   timed={TIMED}", flush=True)

    inference.load_feeder_model(method)
    inference.load_chicken_model(method)

    def one(img):
        t0 = time.perf_counter()
        f = inference.run_feeder(img, method=method)
        t1 = time.perf_counter()
        inference.run_chicken(img, method=method)
        t2 = time.perf_counter()
        ftype = f.feeder_type
        feed_dt = None
        if ftype:
            cv_feeder.measure(img, cv_bundle, ftype, yolo_polygon=f.feeder_polygon, yolo_bbox=f.feeder_bbox)
            feed_dt = time.perf_counter() - t2
        return (t1 - t0), (t2 - t1), feed_dt, (ftype is not None)

    print("warming up…", flush=True)
    for i in range(WARMUP):
        one(imgs[i % len(imgs)])

    seg, det, feed, tot = [], [], [], []
    for i in range(TIMED):
        s, d, fd, had_feeder = one(imgs[i % len(imgs)])
        seg.append(s)
        det.append(d)
        total = s + d + (fd or 0.0)
        tot.append(total)
        if fd is not None:
            feed.append(fd)
        flag = "" if had_feeder else "  (no feeder -> feed skipped)"
        feed_s = f"{fd:.3f}" if fd is not None else "  -  "
        print(f"  cycle {i + 1:2d}: seg={s:.3f}s  det={d:.3f}s  feed={feed_s}s  total={total:.3f}s{flag}", flush=True)

    def line(name, xs):
        if not xs:
            print(f"  {name:6s}  n/a (no feeder detected in the test images)")
            return
        print(f"  {name:6s}  mean={st.mean(xs):.3f}s   min={min(xs):.3f}s   max={max(xs):.3f}s   n={len(xs)}")

    print("\n==== per-cycle compute ({}) ====".format("CUDA " + dev if cuda else "CPU"))
    line("seg", seg)
    line("det", det)
    line("feed", feed)
    line("total", tot)


if __name__ == "__main__":
    main()
