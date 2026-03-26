#!/bin/bash
#
# Installation script for the Multi-Task Legal LLM project.
# Standardizes the environment with all required AI and Data Science libraries.

# Stop on first error
set -e

echo "--- Upgrading pip ---"
pip install --upgrade pip

echo "--- Installing PyTorch Ecosystem ---"
# Defaulting to standard torch; adjust with --index-url if specific CUDA versions are needed
pip install torch torchvision torchaudio


pip install nltk

echo "--- Installing Hugging Face Suite ---"
pip install transformers datasets accelerate peft "huggingface-hub[hf-transfer]"

echo "--- Installing Core Data Science & Utility Libraries ---"
pip install pandas scikit-learn tqdm matplotlib seaborn sympy Pillow networkx

echo "--- Installing Specialized Tools (WandB, OpenAI, NLP Metrics) ---"
pip install wandb protobuf einops timm openai umap-learn rouge

echo ""
echo "--- Installation complete! ---"