#!/bin/bash

# Stops and removes containers, networks, volumes, and images defined in the docker-compose.yml
docker-compose down

# Print out the current Docker status
echo "Current Docker container status:"
docker ps -a
