#!/usr/bin/env python3
"""
Export the LipLearner PyTorch lip encoder to ONNX for Android (ONNX Runtime Mobile).

This mirrors the deployed CoreML `LipEncoder.mlpackage`:
  - Input  `v`:         float32 tensor (1, T, 1, 88, 88), pixel values in [0,1], grayscale.
  - `border` is fixed to all-ones inside the graph (exactly as the CoreML const `border_1_value_0`).
  - Output `embedding`: float32 (1, 500).  L2-normalization is done on the client side (as in iOS).

Usage:
  python tools/export_onnx.py \
      --weights /tmp/LipLearner_pretrained_model.pt \
      --out LipLearner_Android/app/src/main/assets/lip_encoder.onnx
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "pretraining"))

from model import VideoModel  # noqa: E402  (from pretraining/model)


class _Args:
    """Minimal stand-in for the argparse Namespace VideoModel expects."""
    n_dimention = 500


class ExportWrapper(nn.Module):
    """Takes only `v` and injects a constant all-ones border, matching the iOS CoreML model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (B, T, C=1, H=88, W=88)
        b, t = v.shape[0], v.shape[1]
        border = torch.ones(b, t, dtype=v.dtype, device=v.device)
        return self.model(v, border)  # (B, 500)


def load_weights(model: nn.Module, weights_path: str) -> None:
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("video_model", ckpt) if isinstance(ckpt, dict) else ckpt
    model_dict = model.state_dict()
    matched = {k: v for k, v in state.items()
               if k in model_dict and v.size() == model_dict[k].size()}
    missing = [k for k in model_dict if k not in matched]
    print(f"[weights] matched {len(matched)}/{len(model_dict)} params")
    if missing:
        print(f"[weights] MISSING (kept at init): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    model_dict.update(matched)
    model.load_state_dict(model_dict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-frames", type=int, default=29)
    args = ap.parse_args()

    model = VideoModel(_Args())
    load_weights(model, args.weights)
    model.eval()
    wrapper = ExportWrapper(model).eval()

    dummy = torch.rand(1, args.check_frames, 1, 88, 88, dtype=torch.float32)
    with torch.no_grad():
        ref = wrapper(dummy)
    print(f"[torch] output shape {tuple(ref.shape)}  norm={ref.norm().item():.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, args.out,
        input_names=["v"], output_names=["embedding"],
        dynamic_axes={"v": {1: "T"}, "embedding": {0: "B"}},
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    print(f"[onnx] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

    # Parity check against ONNX Runtime.
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"v": dummy.numpy()})[0]
    diff = float(np.abs(ort_out - ref.numpy()).max())
    print(f"[verify] torch-vs-onnxruntime max abs diff = {diff:.3e}  ({'OK' if diff < 1e-3 else 'CHECK!'})")

    # Verify a different length works with the dynamic time axis.
    for t in (10, 45, 128):
        d = torch.rand(1, t, 1, 88, 88, dtype=torch.float32).numpy()
        o = sess.run(None, {"v": d})[0]
        assert o.shape == (1, 500), o.shape
    print("[verify] dynamic time axis OK for T in {10,45,128}")


if __name__ == "__main__":
    main()
