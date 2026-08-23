#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:?set PROJECT_ID}
ZONE=${ZONE:-us-central1-c}
INSTANCE_NAME=${INSTANCE_NAME:-helixserve-g2}
MACHINE_TYPE=${MACHINE_TYPE:-g2-standard-8}
IMAGE_FAMILY=${IMAGE_FAMILY:-common-cu124-ubuntu-2204-nvidia}
IMAGE_PROJECT=${IMAGE_PROJECT:-deeplearning-platform-release}
BOOT_DISK_SIZE=${BOOT_DISK_SIZE:-200GB}

# G2 includes attached L4 GPUs in machine type.
gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --create-disk=auto-delete=yes,boot=yes,image-family="$IMAGE_FAMILY",image-project="$IMAGE_PROJECT",size="$BOOT_DISK_SIZE",type=pd-balanced \
  --metadata=startup-script='#!/bin/bash
set -e
apt-get update
apt-get install -y docker.io git
usermod -aG docker $USER
systemctl enable --now docker
'

echo "Created $INSTANCE_NAME in $ZONE" 
