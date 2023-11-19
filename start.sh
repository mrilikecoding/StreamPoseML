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

# Check if required ports are available
for port in 3000 5001; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $port is already in use, please free up this port and try again."
        exit 1
    else
        echo "Port $port is available."
    fi
done

# If debug mode is on, output Docker and Docker Compose versions
if [[ $1 == "--debug" ]]; then
    echo "Docker version: $(docker --version)"
    echo "Docker Compose version: $(docker-compose --version)"
fi

# Log into Docker Hub
echo "Please enter your Docker Hub credentials:"
docker login

# Pull the images
docker pull mrilikecoding/stream_pose_ml_api:latest
docker pull mrilikecoding/web_ui:latest

# Start the services with Docker Compose
echo "Starting services with Docker Compose..."
DOCKER_COMPOSE_OUTPUT=$(docker-compose up -d 2>&1) || {
    echo "Docker Compose exited with an error. Output was:"
    echo $DOCKER_COMPOSE_OUTPUT
    exit 1
}

sleep 15

# If debug mode is on, output Docker system info, Docker stats, Docker disk usage and Docker Compose logs
if [[ $1 == "--debug" ]]; then
    echo "Docker System Info:"
    docker system info

    echo "Docker Disk Usage:"
    docker system df

    echo "Host Disk Usage:"
    df -h

    echo "Docker stats (for 15 seconds):"
    docker stats --no-stream --format "table {{.Container}},{{.CPUPerc}},{{.MemUsage}}" & 
    STATS_PID=$!
    sleep 15
    if kill -0 $STATS_PID 2>/dev/null; then
        kill $STATS_PID
    else
        echo "Docker stats process already completed."
    fi

    echo "Docker Compose logs:"
    docker-compose logs
fi

# Wait for the services to start
echo "Waiting for the services to start..."

# Loop until the React server is up (may require curl to be installed)
while ! curl -s http://localhost:3000 > /dev/null
do
  echo "Waiting for UI to compile and server to start..."
  sleep 5
done

echo "UI server is up! Launching StreamPoseML application in Browser..."

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
