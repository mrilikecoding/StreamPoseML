#!/bin/bash

# Make sure to rebuild the Docker image after making changes here...
# docker-compose build poser_parser_api

source /venv/bin/activate 
echo "Starting Pose Parser API Server..."
python ./api/app.py