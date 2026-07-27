#!/usr/bin/env python3
"""Dump the mouth ROI crops that the VSR model actually sees, as a contact-sheet PNG.

Usage (from the auto_avsr repo root):
  python dump_crops.py --video /tmp/test_en.mp4 --out /tmp/crops.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image


def read_video_frames(path):
    """Decode to (T,H,W,3) RGB using OpenCV, which applies the container's display
    rotation. Phone clips carry a 90° rotation matrix; PyAV returns the raw sideways
    frames, which makes face/mouth alignment fail completely."""
    import cv2
    cap = cv2.VideoCapture(os.path.abspath(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="/tmp/crops.png")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=40)
    a = ap.parse_args()

    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess

    video = read_video_frames(a.video)
    print(f"[crops] video: {video.shape}  (T,H,W,C)")

    detector = LandmarksDetector()
    landmarks = detector(video)
    detected = sum(1 for l in landmarks if l is not None)
    print(f"[crops] face detected in {detected}/{len(landmarks)} frames")

    vp = VideoProcess(convert_gray=False)
    crops = vp(video, landmarks)
    crops = np.asarray(crops)
    print(f"[crops] ROI array: {crops.shape}")

    # sample evenly, build a contact sheet
    n = min(a.max_frames, len(crops))
    idx = np.linspace(0, len(crops) - 1, n).astype(int)
    sel = crops[idx]
    h, w = sel.shape[1], sel.shape[2]
    cols = a.cols
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(sel):
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img.astype(np.uint8)
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    Image.fromarray(sheet).save(a.out)
    print(f"[crops] wrote {a.out}  ({rows}x{cols} grid of {w}x{h} ROIs)")


if __name__ == "__main__":
    main()
