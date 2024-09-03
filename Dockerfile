FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN date

# ubuntu package install 
RUN apt update && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    python3-pip \
    python3-setuptools \
    wget \
    git \
    tzdata \
    libgl1-mesa-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    liblzma-dev 

RUN apt-get update && \
    apt-get install git-lfs

ARG PYTHON_VERSION="3.10.11"

# Install pyenv
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN eval "$(pyenv init -)" && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION

RUN python3 --version
WORKDIR /app

# Python environment settings
ENV POETRY_HOME="/root/.local" \
    PYTHONUNBUFFERED=1
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry config virtualenvs.in-project true