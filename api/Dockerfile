FROM python:3.11-slim

ARG env=prod
ENV ENV_TYPE=${env}
ARG flask_debug=False
ENV FLASK_DEBUG=${flask_debug}

# for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /usr/src/app

# Copy pyproject.toml, LICENSE, README, and stream_pose_ml source
COPY pyproject.toml LICENSE README.md ./
COPY stream_pose_ml/ ./stream_pose_ml/

# Install stream_pose_ml with dependencies based on environment
RUN if [ "$ENV_TYPE" = "dev" ]; then \
        pip install .[api,dev]; \
    else \
        pip install .[api]; \
    fi

# Copy API code
COPY api/ ./api/

# Set working directory to api
WORKDIR /usr/src/app/api

ENTRYPOINT ["python", "run.py"]