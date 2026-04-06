#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-ncu_cu_seqlens}

ncu \
  --set full \
  --target-processes all \
  --export "$OUT" \
  python3 scripts/profile_cu_seqlens.py

echo "Wrote ${OUT}.ncu-rep"
