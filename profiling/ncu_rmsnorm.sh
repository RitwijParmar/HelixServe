#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-ncu_rmsnorm}

ncu \
  --set full \
  --target-processes all \
  --export "$OUT" \
  python3 -m kernels.benchmark_rmsnorm --rows 4096 --cols 4096

echo "Wrote ${OUT}.ncu-rep"
