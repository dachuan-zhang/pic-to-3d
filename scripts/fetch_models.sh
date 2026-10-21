#!/usr/bin/env bash
# Fetch the ONNX depth model. No torch, no other weights.
# Depth Anything V2 Small (~99 MB) from onnx-community/depth-anything-v2-small.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
out="models/depth_anything_v2_small.onnx"
if [ -f "$out" ]; then echo "$out already present"; exit 0; fi
echo "downloading depth model (~99 MB)…"
curl -fL --progress-bar -o "$out" \
  "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
ls -lh "$out"
