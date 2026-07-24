#!/usr/bin/env bash
# Export the PyTorch lip encoder to ONNX and drop it into the app assets.
#
# Prerequisite: place the pretrained weights at the path below. They are distributed via the
# Google Drive link in the repo README (LipLearner_pretrained_model.pt) and must be downloaded
# manually while signed in to a Google account (the link is gated / rate-limited for anonymous
# access). Then run this script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WEIGHTS="${1:-/tmp/LipLearner_pretrained_model.pt}"
OUT="$REPO_ROOT/LipLearner_Android/app/src/main/assets/lip_encoder.onnx"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: weights not found at $WEIGHTS"
  echo "Download LipLearner_pretrained_model.pt from the Google Drive link in README.md,"
  echo "then re-run:  scripts/build_model.sh /path/to/LipLearner_pretrained_model.pt"
  exit 1
fi

# Use an isolated venv so this doesn't touch the system python.
VENV="/tmp/liplearner_export_venv"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet torch onnx onnxruntime numpy
fi

"$VENV/bin/python" "$REPO_ROOT/tools/export_onnx.py" --weights "$WEIGHTS" --out "$OUT"
echo "Done. Wrote $OUT"
