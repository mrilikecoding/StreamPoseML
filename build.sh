#!/bin/bash
docker buildx create --name multiplat_builder --driver docker-container --bootstrap
docker buildx use multiplat_builder

# Building stream_pose_ml_api
echo "Building stream_pose_ml_api for multiple platforms..."
docker buildx build --platform linux/amd64,linux/arm64 -t mrilikecoding/stream_pose_ml_api:latest --push ./api

# Building web_ui
echo "Building web_ui for multiple platforms..."
docker buildx build --platform linux/amd64,linux/arm64 -t mrilikecoding/stream_pose_ml_web_ui:latest --push ./web_ui

# Building mlflow
echo "Building mlflow for multiple platforms..."
docker buildx build --platform linux/amd64,linux/arm64 -t mrilikecoding/stream_pose_ml_mlflow:latest --push ./mlflow

echo "All images have been built and pushed successfully!"
