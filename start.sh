#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null
then
    echo "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Log into Docker Hub
echo "Please enter your Docker Hub credentials:"
docker login

# Pull the images
docker pull mrilikecoding/pose_parser_api:latest
docker pull mrilikecoding/web_ui:latest

# Start the services with Docker Compose
docker-compose up -d && sleep 15 &

# Wait for the services to start
echo "Waiting 15 seconds for the services to start..."
sleep 15 
echo "Launching Poser in Browswer..."

# Check the operating system and open the URL in the browser
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:3000
elif [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:3000
elif [[ "$OSTYPE" == "cygwin" ]]; then
    cygstart http://localhost:3000
elif [[ "$OSTYPE" == "msys" ]]; then
    start http://localhost:3000
elif [[ "$OSTYPE" == "win32" ]]; then
    start http://localhost:3000
else
    echo "Could not detect the operating system to open the browser, please manually open http://localhost:3000 in your browser."
fi
