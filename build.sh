#!/bin/bash
docker buildx create --name multiplat_builder --driver docker-container --bootstrap
docker buildx use multiplat_builder

# Building pose_parser_api
echo "Building pose_parser_api for multiple platforms..."
docker buildx build --platform linux/amd64,linux/arm64 -t mrilikecoding/pose_parser_api:latest --push ./pose_parser

# Building web_ui
echo "Building web_ui for multiple platforms..."
docker buildx build --platform linux/amd64,linux/arm64 -t mrilikecoding/web_ui:latest --push ./web_ui

echo "All images have been built and pushed successfully!"
