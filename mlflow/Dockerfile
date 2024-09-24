# Use Python base image
FROM python:3.10-slim

# Install necessary packages
RUN pip install --no-cache-dir mlflow flask requests

# Expose port 5000 for the Flask API
EXPOSE 5002

# Copy the application code
COPY app.py /app.py

# Set the entrypoint
ENTRYPOINT ["python", "/app.py"]