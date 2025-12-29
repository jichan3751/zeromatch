#!/bin/bash
set -e
source "$HOME/miniconda3/etc/profile.d/conda.sh"

SCRATCH_PATH="../scratch"
ENV_PATH="$SCRATCH_PATH/env_zeromatch"

## script for setting up conda environment.

echo installing conda environment in $ENV_PATH..

conda create -y --prefix $ENV_PATH python=3.8 pip
conda activate $ENV_PATH

pip install uv
uv pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

conda deactivate
