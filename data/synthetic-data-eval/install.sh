#!/bin/bash
set -e

echo "--- 1. Upgrading Pip ---"
pip install --upgrade pip

echo "--- 2. Installing PyTorch 2.6+ for H100 (CUDA 12.1/12.4 compatibility) ---"
# H100s require modern torch versions for optimum performance (Hopper architecture)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124



echo "--- 3. Installing Project Dependencies ---"
pip install transformers datasets accelerate peft evaluate
pip install pandas scikit-learn tqdm matplotlib seaborn networkx pydot
pip install wandb rouge_score

echo "--- 4. Verification ---"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# rouge
pip install rouge rouge-score

