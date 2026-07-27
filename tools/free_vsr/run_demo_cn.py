#!/usr/bin/env python3
"""Chinese free-form VSR demo (CNVSRC baseline), printing BOTH decodings:
   1) full = attention decoder + CTC + beam search (accurate)
   2) ctc  = CTC greedy, single forward pass (on-device / NPU friendly)

Mouth ROIs are produced with the mediapipe pipeline vendored in auto_avsr (already validated),
instead of CNVSRC's RetinaFace stack, so no extra detector weights are needed.

  python run_demo_cn.py --video /tmp/test_cn.mp4 --ckpt /tmp/cnvsrc_models/model_avg_last5_cncvs_cnvsrc-multi.pth
"""
import argparse
import os
import sys
import time

CNVSRC_ROOT = os.path.dirname(os.path.abspath(__file__))
AUTO_AVSR_ROOT = "/tmp/auto_avsr"
sys.path.insert(0, CNVSRC_ROOT)

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf


def read_video_frames(path):
    """OpenCV decode (applies the container's display rotation)."""
    import cv2
    cap = cv2.VideoCapture(os.path.abspath(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def crop_mouths(video):
    """Reuse auto_avsr's mediapipe landmark + mouth-ROI pipeline (96x96 RGB)."""
    saved = list(sys.path)
    sys.path.insert(0, AUTO_AVSR_ROOT)
    try:
        from preparation.detectors.mediapipe.detector import LandmarksDetector
        from preparation.detectors.mediapipe.video_process import VideoProcess
        landmarks = LandmarksDetector()(video)
        detected = sum(1 for l in landmarks if l is not None)
        print(f"[cn-demo] face detected in {detected}/{len(landmarks)} frames")
        rois = VideoProcess(convert_gray=False)(video, landmarks)
    finally:
        sys.path[:] = saved
    return np.asarray(rois)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=os.path.join(CNVSRC_ROOT, "conf", "test_cnvsrc-multi.yaml"))
    a = ap.parse_args()

    from datamodule.transforms import TextTransform
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from espnet.asr.asr_utils import add_results_to_json
    from predict import get_beam_search_decoder, video_pipeline

    cfg = OmegaConf.load(a.config)
    text_transform = TextTransform()
    token_list = text_transform.token_list
    print(f"[cn-demo] vocab = {len(token_list)}")

    model = E2E(len(token_list), cfg.model.visual_backbone)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model.eval()

    video = read_video_frames(a.video)
    print(f"[cn-demo] video {video.shape}")
    rois = crop_mouths(video)
    # (T,H,W,C) -> (T,C,H,W); CNVSRC's pipeline does /255, CenterCrop(88), gray, normalize
    sample = video_pipeline(torch.tensor(rois).permute(0, 3, 1, 2).contiguous().float())
    print(f"[cn-demo] model input {tuple(sample.shape)}")

    with torch.no_grad():
        t0 = time.time()
        enc_feat, _ = model.encoder(sample.unsqueeze(0), None)
        enc_feat = enc_feat.squeeze(0)
        t_enc = time.time() - t0

        # --- CTC greedy (single pass, on-device friendly)
        t0 = time.time()
        ids = model.ctc.argmax(enc_feat.unsqueeze(0)).squeeze(0).tolist()
        collapsed, prev = [], -1
        for i in ids:
            if i != prev:
                if i != 0:
                    collapsed.append(i)
                prev = i
        ctc_text = "".join(token_list[i] for i in collapsed).replace("▁", " ").strip()
        t_ctc = time.time() - t0

        # --- full decode (attention + CTC + beam search)
        t0 = time.time()
        beam_search = get_beam_search_decoder(model, token_list, ctc_weight=0.3)
        nbest = beam_search(enc_feat)
        nbest = [h.asdict() for h in nbest[: min(len(nbest), 1)]]
        full_text = add_results_to_json(nbest, token_list).replace("▁", " ").strip().replace("<eos>", "")
        t_full = time.time() - t0

    print("\n================= RESULT (ZH) =================")
    print(f"[FULL decode | attn+CTC+beam] ({t_enc + t_full:.1f}s)\n  {full_text}\n")
    print(f"[CTC  greedy | single pass  ] ({t_enc + t_ctc:.1f}s)\n  {ctc_text}")
    print("===============================================")


if __name__ == "__main__":
    main()
