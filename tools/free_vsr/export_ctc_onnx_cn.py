#!/usr/bin/env python3
"""Export the CNVSRC (Chinese) encoder + CTC head to ONNX for on-device free-form VSR.

Same contract as the English export:
    video (1, T, 1, 88, 88) in [0,1]  ->  ctc_logits (1, T, 5906)
The attention decoder and beam search are dropped; the client decodes with CTC greedy.

Note: CNVSRC's E2E feeds the video straight into `encoder` (no separate frontend/proj_encoder),
unlike auto_avsr.

Usage (from the CNVSRC repo root):
  python export_ctc_onnx_cn.py \
      --ckpt /tmp/cnvsrc_models/model_avg_last5_cncvs_cnvsrc-multi.pth \
      --out /tmp/cnvsrc_ctc.onnx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from omegaconf import OmegaConf


class CTCEncoder(nn.Module):
    """encoder -> ctc_lo, i.e. one static forward pass, no autoregressive decoding."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.ctc_lo = model.ctc.ctc_lo

    def forward(self, v):
        # v: (B, T, 1, 88, 88) — the layout CNVSRC's video_pipeline produces
        x, _ = self.encoder(v, None)
        return self.ctc_lo(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="conf/test_cnvsrc-multi.yaml")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from datamodule.transforms import TextTransform
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

    cfg = OmegaConf.load(args.config)
    tokens = TextTransform().token_list
    print(f"[export] vocab = {len(tokens)}")

    model = E2E(len(tokens), cfg.model.visual_backbone)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    net = CTCEncoder(model).eval()
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
    print(f"[onnx] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"v": dummy.numpy()})[0]
    diff = float(np.abs(o - ref.numpy()).max())
    print(f"[verify] torch-vs-ort max diff = {diff:.3e} ({'OK' if diff < 1e-2 else 'CHECK'})")
    for t in (30, 90):
        d = torch.randn(1, t, 1, 88, 88).numpy()
        print(f"[verify] T={t} -> {sess.run(None, {'v': d})[0].shape}")

    # Token list for the client-side CTC decoder.
    tok_out = os.path.splitext(args.out)[0] + "_tokens.txt"
    with open(tok_out, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")
    print(f"[tokens] wrote {tok_out} ({len(tokens)} entries)")


if __name__ == "__main__":
    main()
