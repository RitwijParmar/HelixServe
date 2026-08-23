#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-helixserve_trace}

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=nvtx \
  --capture-range-end=stop \
  --output "$OUT" \
  python3 scripts/profile_decode.py


echo "Wrote ${OUT}.qdrep"
