#!/bin/bash
set -e

# Install OpenSlide if not present
if ! brew list openslide &>/dev/null; then
    echo "Installing openslide..."
    brew install openslide
else
    echo "openslide already installed."
fi

# Create Conda environment
if ! conda env list | grep -q "hovernet_env"; then
    echo "Creating Conda environment 'hovernet_env'..."
    conda create -y -n hovernet_env python=3.10
else
    echo "Conda environment 'hovernet_env' already exists."
fi

# Initialize conda for shell interaction
eval "$(conda shell.bash hook)"
conda activate hovernet_env

# Install PyTorch and dependencies
echo "Installing PyTorch..."
# Using pip for PyTorch on Mac usually works best for getting the MPS enabled version if available, 
# or standard CPU. For verification, we just need it to run.
pip install torch torchvision torchaudio

echo "Installing other dependencies..."
pip install -r requirements_modern.txt
pip install gdown

echo "Environment setup complete."
