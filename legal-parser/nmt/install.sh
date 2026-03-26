#!/bin/bash
#
# Installation script for the Multi-Task Legal LLM project.
# This script installs all necessary Python libraries using pip.
#
# It is strongly recommended to run this inside a Python virtual environment.
# To create and activate a virtual environment (on Linux/macOS):
#   python3 -m venv .venv
#   source .venv/bin/activate
#
# To run this script:
#   bash install_requirements.sh
#

# --- Stop on first error ---
set -e

echo "--- Upgrading pip ---"
pip install --upgrade pip

pip install seaborn

# --- Install PyTorch with CUDA support ---
# The specific command can depend on your system's CUDA version.
# Visit https://pytorch.org/get-started/locally/ to get the precise command for your setup.
# The following command is for CUDA 12.1, which is common for modern GPUs.
echo "--- Installing PyTorch, TorchVision, and Torchaudio ---"
pip install torch torchvision torchaudio 

# --- Install Hugging Face Libraries ---
# `transformers`: For loading models and tokenizers.
# `datasets`: Used in some of your data loading classes for efficient processing.
# `accelerate`: Required for advanced training features like FSDP.
echo "--- Installing Hugging Face libraries (transformers, datasets, accelerate) ---"
pip install transformers datasets accelerate hf_xe
# --- Install Core Data Science and Utility Libraries ---
# `pandas`: For data manipulation (e.g., in LegalClassificationDataset).
# `scikit-learn`: For metrics (f1_score) and data splitting (train_test_split).
# `tqdm`: For creating progress bars.
# `matplotlib`: For plotting the training history.
# `sympy`: Used in your evaluation script for symbolic math comparisons.
# `Pillow`: For image processing (used in an earlier version of your evaluation script).
# `networkx`: For graph construction and manipulation.
echo "--- Installing data science and utility libraries ---"
pip install pandas scikit-learn tqdm matplotlib sympy Pillow networkx

# --- Install NLP and Machine Learning Libraries ---
# `nltk`: For natural language processing tasks.
# `sentence-transformers`: For sentence embeddings.
# `openai`: For OpenAI API interactions.
# `sentencepiece`: For tokenization in some transformer models.
echo "--- Installing NLP and machine learning libraries (nltk, sentence-transformers, openai, sentencepiece) ---"
pip install nltk sentence-transformers openai sentencepiece
pip install sacremoses
echo ""
echo "--- Installation complete! ---"
echo "You can now run your training and evaluation scripts."