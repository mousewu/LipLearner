#!/usr/bin/env python3
"""English VSR demo for auto_avsr (CPU/mediapipe). Prints BOTH decodings:
   1) full  = attention decoder + CTC + beam search (accurate, NOT on-device friendly)
   2) ctc   = CTC greedy, single forward pass (on-device / NPU friendly, lower accuracy)

Usage (from the auto_avsr repo root):
  python run_demo.py --video /path/clip.mp4 --ckpt /tmp/avsr_vsr_base.pth
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch


def read_video_frames(path):
    """Decode an mp4 to (T, H, W, 3) uint8 RGB via OpenCV.
    OpenCV applies the container's display rotation — phone clips carry a 90°
    rotation matrix, and feeding the raw sideways frames makes mouth alignment fail.
    (torchvision.io.read_video was removed in recent torchvision.)"""
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


def build(ckpt_path):
    from lightning import ModelModule
    from datamodule.transforms import VideoTransform
    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess

    args = argparse.Namespace(modality="video", pretrained_model_path=None, ctc_weight=0.1)
    mm = ModelModule(args)
    mm.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    mm.eval()
    return mm, LandmarksDetector(), VideoProcess(convert_gray=False), VideoTransform(subset="test")


def preprocess(video_path, detector, video_process, video_transform):
    video = read_video_frames(video_path)
    print(f"[demo] {len(video)} frames read", flush=True)
    landmarks = detector(video)
    video = video_process(video, landmarks)
    video = torch.tensor(video).permute((0, 3, 1, 2))
    return video_transform(video)


def ctc_greedy(mm, sample):
    with torch.no_grad():
        x = mm.model.frontend(sample.unsqueeze(0))
        x = mm.model.proj_encoder(x)
        enc, _ = mm.model.encoder(x, None)
        ids = mm.model.ctc.argmax(enc).squeeze(0).tolist()  # per-frame argmax, blank=0
    collapsed, prev = [], -1
    for i in ids:
        if i != prev:
            if i != 0:
                collapsed.append(i)
            prev = i
    return mm.text_transform.post_process(torch.tensor(collapsed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()

    print("[demo] loading model…", flush=True)
    mm, detector, video_process, video_transform = build(a.ckpt)
    sample = preprocess(a.video, detector, video_process, video_transform)

    t0 = time.time()
    full = mm(sample)
    t1 = time.time()
    ctc = ctc_greedy(mm, sample)
    t2 = time.time()

    print("\n================= RESULT (EN) =================")
    print(f"[FULL  decode | attn+CTC+beam] ({t1 - t0:.1f}s)\n  {full}\n")
    print(f"[CTC   greedy  | single pass ]  ({t2 - t1:.1f}s)\n  {ctc}")
    print("===============================================")


if __name__ == "__main__":
    main()
