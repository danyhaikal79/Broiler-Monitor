"""
Video utilities: sample frames from an uploaded clip for analysis.

A video is many frames; we sample a capped number evenly across the timeline
so the dashboard runs the pipeline a bounded number of times.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def is_video(filename: str) -> bool:
    return filename.lower().endswith(VIDEO_EXTS)


def capture_frame(index: int = 0, width: int = 1280, height: int = 720):
    """
    Grab ONE frame from the camera attached to the machine running this app.
    Works on the local PC (its webcam) and on the Jetson (its camera) identically.
    Returns a PIL.Image, or None if the camera can't be read.
    """
    import platform
    import cv2

    # Windows webcams are flaky on the default MSMF backend -> use DirectShow.
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = None, None
        for _ in range(3):  # discard the first warm-up frames
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def sample_frames(video_bytes: bytes, max_frames: int = 24) -> list[tuple[float, Image.Image]]:
    """Return up to `max_frames` (timestamp_sec, PIL.Image) pairs sampled evenly."""
    import cv2

    # cv2 needs a file path; write the upload to a temp file (closed before open on Windows)
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(video_bytes)
        path = tf.name

    out: list[tuple[float, Image.Image]] = []
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total > 0:
            idxs = np.linspace(0, total - 1, min(max_frames, total)).astype(int)
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.append((idx / fps, Image.fromarray(rgb)))
        else:
            # Unknown frame count -> read sequentially
            while len(out) < max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.append((len(out) / fps, Image.fromarray(rgb)))
        cap.release()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    return out
