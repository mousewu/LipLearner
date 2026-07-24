#!/usr/bin/env python3
"""
Export mpc001 word-level lipreading checkpoints (Lipreading_using_Temporal_Convolutional_Networks)
to ONNX with the SAME client contract as the LipLearner encoder:

  input  `v`         : float32 (1, T, 1, 88, 88), grayscale pixels in [0,1]
  output `embedding` : float32 (1, 500)  -- the 500-class LRW logits, used as an embedding
                       (L2-normalized on the client, like iOS/Android LipEncoder).

The model's own preprocessing (x/255 then Normalize(0.421, 0.165)) is baked into the graph so the
Android side keeps feeding plain [0,1] grayscale for every model. Temporal consensus is replaced by
a plain mean-over-time so the time axis stays dynamic.

Requires the mpc001 repo checked out (default /tmp/mpc001_tcn).

Usage:
  python tools/export_mpc001_onnx.py --arch snv05x_tcn1x \
      --weights /tmp/lipmodels/snv05x_tcn1x.pth \
      --out LipLearner_Android/app/src/main/assets/mpc001_snv05x_tcn1x.onnx
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

MPC_ROOT = os.environ.get("MPC_ROOT", "/tmp/mpc001_tcn")
sys.path.insert(0, MPC_ROOT)

from lipreading.model import Lipreading  # noqa: E402

# Per-architecture config, mirroring configs/lrw_*.json in the mpc001 repo.
ARCHS = {
    "snv05x_tcn1x":   dict(backbone_type="shufflenet", relu_type="prelu", width_mult=0.5,
                           kernel=[3], layers=4, tcn_width=1, dwpw=False),
    "snv1x_tcn1x":    dict(backbone_type="shufflenet", relu_type="prelu", width_mult=1.0,
                           kernel=[3], layers=4, tcn_width=1, dwpw=False),
    "snv1x_dsmstcn3x":dict(backbone_type="shufflenet", relu_type="relu", width_mult=1.0,
                           kernel=[3, 5, 7], layers=4, tcn_width=1, dwpw=True),
    "resnet18_mstcn": dict(backbone_type="resnet", relu_type="swish", width_mult=1.0,
                           kernel=[3, 5, 7], layers=4, tcn_width=1, dwpw=False),
}

MEAN, STD = 0.421, 0.165


def build_model(cfg):
    hidden_dim = 256
    tcn_options = {
        "num_layers": cfg["layers"],
        "kernel_size": cfg["kernel"],
        "dropout": 0.2,
        "dwpw": cfg["dwpw"],
        "width_mult": cfg["tcn_width"],
    }
    model = Lipreading(
        modality="video",
        num_classes=500,
        tcn_options=tcn_options,
        densetcn_options={},
        backbone_type=cfg["backbone_type"],
        relu_type=cfg["relu_type"],
        width_mult=cfg["width_mult"],
        use_boundary=False,
        extract_feats=False,
    )
    return model


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:5])


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Replace length-based temporal consensus with a plain mean over the time axis so ONNX keeps
        # a dynamic T.  Both TCN and MultiscaleMultibranchTCN expose .consensus_func.
        self.model.tcn.consensus_func = lambda x, lengths, B: x.mean(dim=2)

    def forward(self, v):
        # v: (B, T, 1, 88, 88) in [0,1]
        v = (v - MEAN) / STD
        x = v.permute(0, 2, 1, 3, 4)          # (B, 1, T, 88, 88) -> (B, C, T, H, W)
        return self.model(x, None, None)       # (B, 500) logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCHS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = build_model(ARCHS[args.arch])
    load_ckpt(model, args.weights)
    model.eval()
    wrapper = ExportWrapper(model).eval()

    # mpc001 models were trained on fixed 29-frame clips, and the ShuffleNet channel-shuffle bakes a
    # fixed batch dim into a Reshape — so we export a FIXED 29-frame graph. The Android client
    # resamples every clip to 29 frames for these models.
    FRAMES = 29
    dummy = torch.rand(1, FRAMES, 1, 88, 88, dtype=torch.float32)
    with torch.no_grad():
        ref = wrapper(dummy)
    print(f"[torch] {args.arch} output {tuple(ref.shape)} norm={ref.norm().item():.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, args.out,
        input_names=["v"], output_names=["embedding"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    print(f"[onnx] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"v": dummy.numpy()})[0]
    diff = float(np.abs(o - ref.numpy()).max())
    print(f"[verify] torch-vs-ort max diff = {diff:.3e}  ({'OK' if diff < 1e-3 else 'CHECK'})")
    assert o.shape == (1, 500)
    print(f"[verify] fixed {FRAMES}-frame graph OK")


if __name__ == "__main__":
    main()
