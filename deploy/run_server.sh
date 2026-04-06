#!/usr/bin/env bash
set -euo pipefail

export HELIX_USE_TOY_BACKEND=${HELIX_USE_TOY_BACKEND:-1}
export HELIX_DEVICE=${HELIX_DEVICE:-cuda}
export HELIX_PORT=${HELIX_PORT:-8000}

python3 -m uvicorn server.main:app --host 0.0.0.0 --port "$HELIX_PORT"
