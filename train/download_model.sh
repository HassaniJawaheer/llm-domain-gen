#!/bin/bash

# Variables
GENERATION_REPO="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
MODEL_PATH="model-based"

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs is not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y git-lfs
    git lfs install
fi

# Download
export GIT_LFS_SKIP_SMUDGE=0
echo "Clonage du modèle depuis $GENERATION_REPO..."
git clone "$GENERATION_REPO" "$MODEL_PATH"
