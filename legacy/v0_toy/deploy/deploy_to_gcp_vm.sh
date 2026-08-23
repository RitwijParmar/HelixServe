#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:?set PROJECT_ID}
ZONE=${ZONE:-us-central1-c}
INSTANCE_NAME=${INSTANCE_NAME:?set INSTANCE_NAME}
REPO=${REPO:-helixserve}
TAG=${TAG:-latest}
IMAGE="gcr.io/${PROJECT_ID}/${REPO}:${TAG}"

# Build and push container.
gcloud builds submit --project="$PROJECT_ID" --tag "$IMAGE" .

# Pull and run container on VM.
gcloud compute ssh "$INSTANCE_NAME" --zone "$ZONE" --project "$PROJECT_ID" --command "\
  sudo docker pull $IMAGE && \
  sudo docker rm -f helixserve || true && \
  sudo docker run --gpus all -d --name helixserve -p 8000:8000 \
    -e HELIX_USE_TOY_BACKEND=0 \
    -e HELIX_MODEL=sshleifer/tiny-gpt2 \
    -e HELIX_DEVICE=cuda \
    $IMAGE\
"

echo "Deployed $IMAGE to $INSTANCE_NAME"
