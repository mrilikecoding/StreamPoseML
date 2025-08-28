#!/bin/bash

# Stops and removes containers, networks, volumes, and images defined in the docker-compose files
# Try to stop services from both compose files to ensure all containers are stopped

echo "Stopping services from docker-compose.build.yml..."
docker compose -f docker-compose.build.yml down 2>/dev/null || echo "No services running from docker-compose.build.yml"

echo "Stopping services from docker-compose.local.yml..."
docker compose -f docker-compose.local.yml down 2>/dev/null || echo "No services running from docker-compose.local.yml"

# Print out the current Docker status
echo "Current Docker container status:"
docker ps -a
