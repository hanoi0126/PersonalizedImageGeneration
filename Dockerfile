# ベースイメージ
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

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
    python3-venv \ 
    openssh-client \
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

# pyenv用の環境変数
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
ENV PYTHON_VERSION="3.10.11"

# ここでまとめて pyenv のインストール & Python バージョン設定 & Poetry のインストールを行う
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash && \
    # 確認 (ログ出力するだけ)
    python --version && which python && \
    # Poetryのインストール
    curl -sSL https://install.python-poetry.org | python -

# Poetry のパスを追加
ENV POETRY_HOME="/root/.local"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Pythonのパスを追加
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# 作業ディレクトリ
WORKDIR /workspace

# pyproject.toml / poetry.lock をコピーして依存関係をインストール
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root
