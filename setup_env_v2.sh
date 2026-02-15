#!/bin/bash
set -e

ENV_PIP="/Users/Hrithik/miniconda3/envs/hovernet_env/bin/pip"

echo "Installing PyTorch..."
$ENV_PIP install torch torchvision torchaudio

echo "Installing other dependencies..."
$ENV_PIP install -r requirements_modern.txt
$ENV_PIP install gdown

echo "Environment setup complete."
