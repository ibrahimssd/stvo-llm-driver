#!/bin/bash

# Configuration for the Tmux Session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_OUT="out/tmux-legal-kge-train-$TIMESTAMP.out"
LOG_ERR="err/tmux-legal-kge-train-$TIMESTAMP.err"

# Ensure log directories exist
mkdir -p out err

# --- PATHS AND PARAMETERS ---
ROOT_DIR="/temp/siddig"
DATA_DIR="$ROOT_DIR/driving-license"


# --- GPU SETUP ---
# Since we are not using Slurm, we manually specify the GPU ID (0-based)
# Use 'nvidia-smi' to find available indices.
export CUDA_VISIBLE_DEVICES=1




# Create and activate the virtual environment if it doesn't already exist
if [ ! -d "$DATA_DIR/envs/driving-env" ]; then
    python3 -m venv $DATA_DIR/envs/driving-env
    source $DATA_DIR/envs/driving-env/bin/activate
    bash ./install.sh
fi 

source $DATA_DIR/envs/driving-env/bin/activate
# bash ./install.sh



# GERMAN DATASETS
python ../legal-KGE/train_legal_kge.py \
    --model_type transe \
    --delimiter "<tr>" \
    --data_path ../data/stvo/de/integrated_Straßenverkehrs_Ordnung_graph_de_triples.txt \
    --embedding_dim 500 \
    --epochs 50 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --margin 1.0\
    --l2_reg 0.01 \
    --dropout 0.1 \
    --scheduler plateau \
    --output_dir ../legal-KGE/embeddings/transe/STvO_de \
    --use_wandb \
    --wandb_project legal-kge-stvo >> "$LOG_OUT" 2>> "$LOG_ERR"

# ENGLISH TRANSLATED DATASETS

# python ../legal-KGE/train_legal_kge.py \
#     --model_type transe \
#     --delimiter "<tr>" \
#     --data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph_triples.txt \
#     --embedding_dim 500 \
#     --epochs 50 \
#     --batch_size 256 \
#     --learning_rate 0.001 \
#     --margin 1.0\
#     --l2_reg 0.01 \
#     --dropout 0.1 \
#     --scheduler plateau \
#     --output_dir ../legal-KGE/embeddings/transe/m2m100_418M_STvO \
#     --use_wandb \
#     --wandb_project legal-kge-stvo >> "$LOG_OUT" 2>> "$LOG_ERR"



# # KG double the dimention complex
# python ../legal-KGE/train_legal_kge.py \
#     --model_type rotate \
#     --delimiter "<tr>" \
#     --data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph_triples.txt \
#     --embedding_dim 250 \
#     --epochs 75 \
#     --batch_size 256 \
#     --learning_rate 0.0001 \
#     --margin 6.0\
#     --l2_reg 0.0001 \
#     --dropout 0.1 \
#     --scheduler cosine \
#     --output_dir ../legal-KGE/embeddings/rotate/m2m100_418M_STvO \
#     --use_wandb \
#     --wandb_project legal-kge-stvo >> "$LOG_OUT" 2>> "$LOG_ERR"


# python ../legal-KGE/train_legal_kge.py \
#     --model_type complex \
#     --delimiter "<tr>" \
#     --data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph_triples.txt \
#     --embedding_dim 250 \
#     --epochs 50 \
#     --batch_size 256 \
#     --learning_rate 0.001 \
#     --l2_reg 0.001 \
#     --dropout 0.1 \
#     --scheduler plateau \
#     --output_dir ../legal-KGE/embeddings/complex/m2m100_418M_STvO \
#     --use_wandb \
#     --wandb_project legal-kge-stvo >> "$LOG_OUT" 2>> "$LOG_ERR"



# python ../legal-KGE/train_legal_kge.py \
#     --model_type  distmult \
#     --delimiter "<tr>" \
#     --data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph_triples.txt \
#     --embedding_dim 500 \
#     --epochs 50 \
#     --batch_size 256 \
#     --learning_rate 0.001 \
#     --l2_reg 0.01 \
#     --dropout 0.1 \
#     --scheduler plateau \
#     --output_dir ../legal-KGE/embeddings/distmult/m2m100_418M_STvO \
#     --use_wandb \
#     --wandb_project legal-kge-stvo >> "$LOG_OUT" 2>> "$LOG_ERR"


# python ../legal-KGE/compare_kge_models.py





# # m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph_triples
# python ../train_transE.py \
#     --data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph_triples.txt \
#     --embedding_dim 500\
#     --epochs 200 \
#     --batch_size 32 \
#     --learning_rate 5e-5 \
#     --l2_reg 0.01 \
#     --dropout 0.1 \
#     --save_paragraph_embeddings \
#     --paragraph_extraction_method regex\
#     --paragraph_prefixes '§'\
#     --paragraph_pattern '^§§?\s*\d+(?:\s*to\s*\d+)?[\s\S]*?(?=\n§|\Z)'\
#     --delimiter "<tr>" \
#     --scheduler cosine \
#     --negative_sampling uniform \
#     --validation_split 0.15 \
#     --early_stopping_patience 15 \
#     --output_dir ../transE_embeddings/m2m100_418M_STvO \



# opus-mt-de-en_translated_integrated_Straßenverkehrs_Ordnung_graph_triples
# python ../train_transE.py \
#     --data_path ../data/stvo/opus-mt-de-en_translated_integrated_Straßenverkehrs_Ordnung_graph_triples.txt \
#     --embedding_dim 500\
#     --epochs 200 \
#     --batch_size 32 \
#     --learning_rate 5e-5 \
#     --l2_reg 0.01 \
#     --dropout 0.1 \
#     --save_paragraph_embeddings \
#     --paragraph_extraction_method regex\
#     --paragraph_prefixes '§'\
#     --paragraph_pattern '^§§?\s*\d+(?:\s*to\s*\d+)?[\s\S]*?(?=\n§|\Z)'\
#     --delimiter "<tr>" \
#     --scheduler cosine \
#     --negative_sampling uniform \
#     --validation_split 0.15 \
#     --early_stopping_patience 15 \
#     --output_dir ../transE_embeddings/opus-mt-de-en_STvO \





    
# # m2m100_418M_translated_BGB_graph_triples
# python ../train_transE.py \
#     --data_path ../data/BGB/m2m100_418M_translated_BGB_graph_triples.txt \
#     --embedding_dim 500\
#     --epochs 200 \
#     --batch_size 32 \
#     --learning_rate 5e-5 \
#     --l2_reg 0.01 \
#     --dropout 0.1 \
#     --save_paragraph_embeddings \
#     --paragraph_extraction_method regex\
#     --paragraph_prefixes '§'\
#     --paragraph_pattern '^§§?\s*\d+(?:\s*to\s*\d+)?[\s\S]*?(?=\n§|\Z)'\
#     --delimiter "<tr>" \
#     --scheduler cosine \
#     --negative_sampling uniform \
#     --validation_split 0.15 \
#     --early_stopping_patience 15 \
#     --output_dir ../transE_embeddings/m2m100_418M_BGB \

# # opus-mt-de-en_translated_integrated_BGB_graph_triples
# python ../train_transE.py \
#     --data_path ../data/BGB/opus-mt-de-en_translated_integrated_BGB_graph_triples.txt \
#     --embedding_dim 500\
#     --epochs 200 \
#     --batch_size 32 \
#     --learning_rate 5e-5 \
#     --l2_reg 0.01 \
#     --dropout 0.1 \
#     --save_paragraph_embeddings \
#     --paragraph_extraction_method regex\
#     --paragraph_prefixes '§'\
#     --paragraph_pattern '^§§?\s*\d+(?:\s*to\s*\d+)?[\s\S]*?(?=\n§|\Z)'\
#     --delimiter "<tr>" \
#     --scheduler cosine \
#     --negative_sampling uniform \
#     --validation_split 0.15 \
#     --early_stopping_patience 15 \
#     --output_dir ../transE_embeddings/opus-mt-de-en_BGB \
