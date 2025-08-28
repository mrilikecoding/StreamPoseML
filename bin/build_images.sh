#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Always use --no-cache for release builds to ensure fresh images
echo "Building with --no-cache to ensure fresh builds..."

# Parse command line arguments for optional cache usage (development only)
USE_CACHE=""
if [[ "$1" == "--use-cache" ]]; then
  USE_CACHE="true"
  echo "Warning: Using cache for builds (development mode)"
else
  USE_CACHE="--no-cache"
fi

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

# Define build contexts and Dockerfiles (fixed paths)
API_CONTEXT="."
API_DOCKERFILE="./api/Dockerfile"
WEB_UI_CONTEXT="./web_ui"
WEB_UI_DOCKERFILE="./web_ui/Dockerfile"
MLFLOW_CONTEXT="./mlflow"
MLFLOW_DOCKERFILE="./mlflow/Dockerfile"

# Function to build and push an image
build_and_push() {
  local image=$1
  local context=$2
  local dockerfile=$3
  echo "Building and pushing $image for platforms $PLATFORMS..."
  docker buildx build \
    --platform $PLATFORMS \
    --push \
    $USE_CACHE \
    -t $image \
    -f $dockerfile \
    $context
}

# Ensure we're using buildx
docker buildx use default 2>/dev/null || docker buildx create --use

# Build and push images
echo "Starting Docker image builds..."
echo "================================"

echo "1/3: Building API image..."
build_and_push $API_IMAGE $API_CONTEXT $API_DOCKERFILE

echo "2/3: Building Web UI image..."
build_and_push $WEB_UI_IMAGE $WEB_UI_CONTEXT $WEB_UI_DOCKERFILE

echo "3/3: Building MLflow image..."
build_and_push $MLFLOW_IMAGE $MLFLOW_CONTEXT $MLFLOW_DOCKERFILE

echo "================================"
echo "All images built and pushed successfully!"
echo ""
echo "Note: Images pushed to DockerHub with :latest tag"
echo "To use these images, run: make start"
