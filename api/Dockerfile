FROM python:3.11-slim

ARG env=prod
ENV ENV_TYPE $env
ARG flask_debug=False
ENV FLASK_DEBUG $flask_debug

# for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /usr/src/app

# Copy requirements files
COPY requirements.txt dev-requirements.txt ./

# Install requirements based on environment
RUN if [ "$ENV_TYPE" = "dev" ]; then \
        pip install -r dev-requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi

# Copy API code
COPY . .

ENTRYPOINT ["python", "run.py"]