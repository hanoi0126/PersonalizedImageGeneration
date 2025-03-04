#!/bin/bash

poetry run python utils/download_sd.py

mkdir -p models/fastcomposer ; cd models/fastcomposer
wget https://huggingface.co/mit-han-lab/fastcomposer/resolve/main/pytorch_model.bin