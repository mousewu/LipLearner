#!/usr/bin/env python3
"""End-to-end check of the exported ONNX (encoder+CTC) on a real clip.

Runs the same preprocessing as the demo, feeds the ONNX graph, decodes with CTC greedy,
and prints the transcript — this is exactly what the Android client will do.

  python test_onnx_ctc.py --video /tmp/test_en.mp4 --onnx /tmp/avsr_ctc.onnx
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch


def read_video_frames(path):
    import cv2
    cap = cv2.VideoCapture(os.path.abspath(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--onnx", required=True)
    a = ap.parse_args()

    from datamodule.transforms import VideoTransform, TextTransform
    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess

    video = read_video_frames(a.video)
    landmarks = LandmarksDetector()(video)
    rois = VideoProcess(convert_gray=False)(video, landmarks)
    sample = VideoTransform(subset="test")(torch.tensor(rois).permute(0, 3, 1, 2))
    print(f"[onnx-test] input {tuple(sample.shape)}")

    import onnxruntime as ort
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    t0 = time.time()
    logits = sess.run(None, {"v": sample.unsqueeze(0).numpy().astype(np.float32)})[0]
    dt = time.time() - t0

    ids = logits[0].argmax(-1).tolist()
    collapsed, prev = [], -1
    for i in ids:
        if i != prev:
            if i != 0:
                collapsed.append(i)
            prev = i
    text = TextTransform().post_process(torch.tensor(collapsed))

    print(f"\n=========== ONNX + CTC greedy ({dt:.2f}s) ===========")
    print(f"  {text}")
    print("====================================================")


if __name__ == "__main__":
    main()
