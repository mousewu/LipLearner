#!/usr/bin/env python3
"""Export auto_avsr's encoder + CTC head to ONNX for on-device (NPU-friendly) free-form VSR.

The exported graph is a SINGLE forward pass:
    video (1, T, 1, 88, 88) in [0,1]  ->  ctc_logits (1, T, 5049)
The autoregressive Transformer decoder and beam search are dropped; the client decodes with
CTC greedy (argmax + collapse repeats + drop blank), which needs no model state.

Usage (from the auto_avsr repo root):
  python export_ctc_onnx.py --ckpt /tmp/avsr_vsr_best.pth --out /tmp/avsr_ctc.onnx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn


class CTCEncoder(nn.Module):
    """frontend -> proj_encoder -> conformer encoder -> ctc_lo (no decoder, no beam search)."""

    def __init__(self, model):
        super().__init__()
        self.frontend = model.frontend
        self.proj_encoder = model.proj_encoder
        self.encoder = model.encoder
        self.ctc_lo = model.ctc.ctc_lo

    def forward(self, v):
        # v: (B, T, 1, 88, 88) — same layout the VideoTransform produces
        x = self.frontend(v)
        x = self.proj_encoder(x)
        x, _ = self.encoder(x, None)
        return self.ctc_lo(x)  # (B, T', odim) raw logits; argmax is done client-side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=60, help="dummy clip length for tracing")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from lightning import ModelModule

    margs = argparse.Namespace(modality="video", pretrained_model_path=None, ctc_weight=0.1)
    mm = ModelModule(margs)
    mm.model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    mm.eval()
    print(f"[export] loaded checkpoint, vocab={len(mm.text_transform.token_list)}")

    net = CTCEncoder(mm.model).eval()

    dummy = torch.randn(1, args.frames, 1, 88, 88)
    with torch.no_grad():
        ref = net(dummy)
    print(f"[torch] logits {tuple(ref.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        net, dummy, args.out,
        input_names=["v"], output_names=["ctc_logits"],
        dynamic_axes={"v": {1: "T"}, "ctc_logits": {1: "Tout"}},
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[onnx] wrote {args.out} ({size_mb:.1f} MB)")

    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"v": dummy.numpy()})[0]
    diff = float(np.abs(o - ref.numpy()).max())
    print(f"[verify] torch-vs-ort max diff = {diff:.3e} ({'OK' if diff < 1e-2 else 'CHECK'})")
    for t in (30, 90):
        d = torch.randn(1, t, 1, 88, 88).numpy()
        out = sess.run(None, {"v": d})[0]
        print(f"[verify] T={t} -> {out.shape}")


if __name__ == "__main__":
    main()
