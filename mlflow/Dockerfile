# Use a base image with Conda installed
FROM continuumio/miniconda3

# Install MLflow and any additional base dependencies
RUN pip install --no-cache-dir mlflow flask requests

# Expose port 5002 for the Flask API
EXPOSE 5002

# Copy the application code
COPY app.py /app.py

# Set the entrypoint
ENTRYPOINT ["python", "/app.py"]