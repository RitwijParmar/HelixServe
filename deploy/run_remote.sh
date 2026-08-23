#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="${REPOSITORY_URL:-https://github.com/RitwijParmar/TickYantra.git}"
BRANCH="${BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-/opt/tickyantra}"

sudo mkdir -p "${INSTALL_DIR}"
sudo chown "$(id -u):$(id -g)" "${INSTALL_DIR}"
if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
  git clone --branch "${BRANCH}" --single-branch "${REPOSITORY_URL}" "${INSTALL_DIR}"
else
  git -C "${INSTALL_DIR}" fetch origin "${BRANCH}"
  git -C "${INSTALL_DIR}" checkout "${BRANCH}"
  git -C "${INSTALL_DIR}" pull --ff-only origin "${BRANCH}"
fi
docker compose -f "${INSTALL_DIR}/deploy/compose.gpu.yaml" up -d --build
