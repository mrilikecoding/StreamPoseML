# Use a base image with Conda installed
FROM continuumio/miniconda3
# Install system dependencies for building Python
RUN apt-get update && apt-get install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  wget \
  curl \
  llvm \
  libncurses5-dev \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libffi-dev \
  liblzma-dev \
  git
# Set environment variables for pyenv (if you want to go down the pyenv route)
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install any necessary packages (e.g., pyenv)
RUN apt-get update && apt-get install -y \
  curl \
  && curl https://pyenv.run | bash \
  && echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc \
  && echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc \
  && echo 'eval "$(pyenv init --path)"' >> ~/.bashrc \
  && echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Install Python 3.11 via pyenv and set as default
RUN pyenv install 3.11 && pyenv global 3.11

# Install MLflow and any additional base dependencies
RUN pip install --no-cache-dir virtualenv mlflow==2.18.0 flask requests debugpy


# Expose port 5002 for the Flask API
EXPOSE 5002

# Copy the application code
COPY app.py /app.py

# Set the entrypoint
ENTRYPOINT ["python", "/app.py"]
