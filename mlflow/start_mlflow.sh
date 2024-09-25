#!/bin/bash

# Get the model path from the environment variable or default to /model
MODEL_PATH=${MODEL_PATH:-/model}

# Check if the model path exists
if [ ! -f "$MODEL_PATH/MLmodel" ]; then
  echo "Model not found at $MODEL_PATH. Please ensure the model is available."
  exit 1
fi

echo "Starting MLflow model serving..."

# Serve the model
mlflow models serve -m "$MODEL_PATH" -h 0.0.0.0 -p 5000