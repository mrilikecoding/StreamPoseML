FROM python:3.10-slim

# Default value; will be overridden by build-args, if passed
ARG flask_debug=False

ENV FLASK_DEBUG $flask_debug

# for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY .  ./

ENTRYPOINT ["python", "run.py"]