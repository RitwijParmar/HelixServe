#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y docker.io docker-compose-v2 git curl
systemctl enable --now docker

# Deep Learning VM images include NVIDIA drivers/toolkit. Fail visibly if not usable.
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Hard cost guardrail: stop the VM after two hours even if a benchmark fails.
systemd-run --unit=helix-cost-guard --on-active=2h /sbin/shutdown -h now
