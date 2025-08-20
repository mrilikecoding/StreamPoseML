#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Detect host architecture
HOST_ARCH=$(uname -m)

if [[ "$HOST_ARCH" == "arm64" ]]; then
  # Apple Silicon: prioritize ARM64
  PLATFORMS="linux/arm64"
else
  # Default to Linux/AMD64
  PLATFORMS="linux/amd64,linux/arm64"
fi

# Define image tags
API_IMAGE="mrilikecoding/stream_pose_ml_api:latest"
WEB_UI_IMAGE="mrilikecoding/stream_pose_ml_web_ui:latest"
MLFLOW_IMAGE="mrilikecoding/stream_pose_ml_mlflow:latest"

# Define build contexts
API_CONTEXT="."
WEB_UI_CONTEXT="./web_ui"
MLFLOW_CONTEXT="./mlflow"

# Function to build and push an image
build_and_push() {
  local image=$1
  local context=$2
  local dockerfile=$3
  echo "Building and pushing $image for platforms $PLATFORMS..."
  docker buildx build \
    --platform $PLATFORMS \
    --push \
    -t $image \
    -f $dockerfile \
    $context
}

# Build and push images
build_and_push $API_IMAGE $API_CONTEXT "./api/Dockerfile"
build_and_push $WEB_UI_IMAGE $WEB_UI_CONTEXT "Dockerfile"
build_and_push $MLFLOW_IMAGE $MLFLOW_CONTEXT "Dockerfile"

echo "All images built and pushed successfully!"
