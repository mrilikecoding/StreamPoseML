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

# Extract version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
if [[ -z "$VERSION" ]]; then
  echo "Error: Could not extract version from pyproject.toml"
  exit 1
fi
echo "Building images for version: $VERSION"

# Define image tags (both latest and version)
API_IMAGE_BASE="mrilikecoding/stream_pose_ml_api"
WEB_UI_IMAGE_BASE="mrilikecoding/stream_pose_ml_web_ui"
MLFLOW_IMAGE_BASE="mrilikecoding/stream_pose_ml_mlflow"

API_TAGS="$API_IMAGE_BASE:latest $API_IMAGE_BASE:v$VERSION"
WEB_UI_TAGS="$WEB_UI_IMAGE_BASE:latest $WEB_UI_IMAGE_BASE:v$VERSION"
MLFLOW_TAGS="$MLFLOW_IMAGE_BASE:latest $MLFLOW_IMAGE_BASE:v$VERSION"

# Define build contexts and Dockerfiles (fixed paths)
API_CONTEXT="."
API_DOCKERFILE="./api/Dockerfile"
WEB_UI_CONTEXT="./web_ui"
WEB_UI_DOCKERFILE="./web_ui/Dockerfile"
MLFLOW_CONTEXT="./mlflow"
MLFLOW_DOCKERFILE="./mlflow/Dockerfile"

# Function to build and push an image with multiple tags
build_and_push() {
  local tags=$1
  local context=$2
  local dockerfile=$3
  local image_name=$(echo $tags | cut -d' ' -f1 | cut -d':' -f1)
  echo "Building and pushing $image_name for platforms $PLATFORMS..."
  echo "Tags: $tags"
  
  # Build tag arguments
  local tag_args=""
  for tag in $tags; do
    tag_args="$tag_args -t $tag"
  done
  
  docker buildx build \
    --platform $PLATFORMS \
    --push \
    $USE_CACHE \
    $tag_args \
    -f $dockerfile \
    $context
}

# Ensure we're using buildx
docker buildx use default 2>/dev/null || docker buildx create --use

# Build and push images
echo "Starting Docker image builds..."
echo "================================"

echo "1/3: Building API image..."
build_and_push "$API_TAGS" $API_CONTEXT $API_DOCKERFILE

echo "2/3: Building Web UI image..."
build_and_push "$WEB_UI_TAGS" $WEB_UI_CONTEXT $WEB_UI_DOCKERFILE

echo "3/3: Building MLflow image..."
build_and_push "$MLFLOW_TAGS" $MLFLOW_CONTEXT $MLFLOW_DOCKERFILE

echo "================================"
echo "All images built and pushed successfully!"
echo ""
echo "Images pushed to DockerHub with tags:"
echo "  :latest (for make start)"  
echo "  :v$VERSION (for version pinning)"
echo ""
echo "To use latest images: make start"
echo "To use specific version: docker pull mrilikecoding/stream_pose_ml_api:v$VERSION"
