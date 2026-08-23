#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-a32b73cc-f720-4252-993}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-tickyantra-l4}"

gcloud services enable compute.googleapis.com --project "${PROJECT_ID}"
if ! gcloud compute instances describe "${INSTANCE_NAME}" --zone "${ZONE}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute instances create "${INSTANCE_NAME}" \
    --project "${PROJECT_ID}" \
    --zone "${ZONE}" \
    --machine-type g2-standard-8 \
    --maintenance-policy TERMINATE \
    --provisioning-model STANDARD \
    --restart-on-failure \
    --boot-disk-size 100GB \
    --boot-disk-type pd-balanced \
    --image-family common-cu129-ubuntu-2204-nvidia-580 \
    --image-project deeplearning-platform-release \
    --metadata-from-file startup-script=deploy/gcp_startup.sh
fi

echo "Created ${INSTANCE_NAME} in ${ZONE}. Access it with IAP; no public inference port is exposed."
