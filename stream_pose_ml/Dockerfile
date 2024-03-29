FROM python:3.10-slim

# Default value; will be overridden by build-args, if passed
ARG flask_debug=False

ENV FLASK_DEBUG $flask_debug

# for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /usr/src/app

COPY .  ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir --user -r requirements.txt

ENTRYPOINT ["python", "run.py"]