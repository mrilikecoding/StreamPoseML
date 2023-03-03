FROM python:3.10

# needed for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /usr/src/app

COPY .  ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir --user -r requirements.txt