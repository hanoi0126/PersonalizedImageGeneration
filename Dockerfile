# ベースイメージの指定
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 環境変数設定とタイムゾーンの設定
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    tmux \
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
    libsqlite3-dev \
    liblzma-dev \
    libglib2.0-0 \
    git-lfs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 

# pyenvインストールおよびPythonバージョンの設定
ARG PYENV_ROOT="/root/.pyenv"
ARG PYTHON_VERSION="3.10.11"

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    export PATH="$PYENV_ROOT/bin:$PATH" && \
    eval "$(pyenv init --path)" && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash

# Poetryのインストール
ENV POETRY_HOME="/root/.local" \
    PATH="$POETRY_HOME/bin:$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Pythonのパスを追加
ENV PYTHONPATH="/workspace:$PYTHONPATH" \
    PATH="/root/.local/bin:$PATH"

# プロジェクトファイルをコピーしてPoetryで依存関係をインストール
WORKDIR /workspace
COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root
