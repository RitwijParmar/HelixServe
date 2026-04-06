#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-ncu_rmsnorm}

ncu \
  --set full \
  --target-processes all \
  --export "$OUT" \
  python3 scripts/profile_rmsnorm.py

echo "Wrote ${OUT}.ncu-rep"
