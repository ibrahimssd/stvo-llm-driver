#!/bin/bash
# Configuration for the Tmux Session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_OUT="out/tmux-STVO-train-$TIMESTAMP.out"
LOG_ERR="err/tmux-STVO-train-$TIMESTAMP.err"

# Ensure log directories exist
mkdir -p out err

# --- PATHS AND PARAMETERS ---
ROOT_DIR="/temp/siddig"
DATA_DIR="$ROOT_DIR/driving-license"
HUGG_OUT_DIR="$DATA_DIR/STVo_PRLM_HF_hierarchical_models_20_epochs_10_segments"
OUT_DIR="$DATA_DIR/STVo_PRLM_PY_hierarchical_models_20_epochs_10_segments"
HF_DIR="$DATA_DIR/HF_models"

mkdir -p $DATA_DIR 

# --- GPU SETUP ---
# Since we are not using Slurm, we manually specify the GPU ID (0-based)
# Use 'nvidia-smi' to find available indices.

GPU_ID=1  # Change this to the appropriate GPU ID if you have multiple GPUs

# --- ENVIRONMENT VARIABLES ---
export HF_HOME=$HF_DIR
export TRANSFORMERS_CACHE=$HF_DIR
export HF_DATASETS_CACHE=$HF_DIR
export MPLCONFIGDIR="$DATA_DIR/.matplotlib"
export WANDB_DIR="$DATA_DIR/wandb_logs"
export WANDB_CACHE_DIR="$DATA_DIR/wandb_cache"
export WANDB_DATA_DIR="$DATA_DIR/wandb_data"


# Log GPU Info
mkdir -p ./MIG-IDs
SLURM_JOB_ID="TMUX_$TIMESTAMP"
echo "$SLURM_JOB_ID, $CUDA_VISIBLE_DEVICES" > ./MIG-IDs/$SLURM_JOB_ID.csv
nvidia-smi -L >> ./MIG-IDs/$SLURM_JOB_ID.csv

# Create and activate the virtual environment if it doesn't already exist
if [ ! -d "$DATA_DIR/envs/driving-env" ]; then
    python3 -m venv $DATA_DIR/envs/driving-env
    source $DATA_DIR/envs/driving-env/bin/activate
    bash ./install.sh
fi 

source $DATA_DIR/envs/driving-env/bin/activate
# bash ./install.sh
     

MASKED_MODELS=(
    
    # nlpaueb/bert-base-uncased-eurlex
    # nlpaueb/bert-base-uncased-contracts
    nlpaueb/bert-base-uncased-echr
    # casehold/custom-legalbert
    # dlicari/Italian-Legal-BERT   
    # google-bert/bert-base-uncased
    # nlpaueb/legal-bert-base-uncased
    # nlpaueb/legal-bert-small-uncased
    # avichr/Legal-heBERT
    
    # microsoft/deberta-base #exceeding memory
) 

# CAUSAL_MODELS=(
#     Qwen/Qwen3-Embedding-0.6B
#     google/gemma-3-270m
#     google/gemma-3-270m-it
#     openai-community/gpt2
# )


###################### STvO with m2m100_418M translation ##########################
# nsp_clu_cls : 0.3 , 0.3 , 0.4
# # transe works perfect 
# out_DIR_TRanse="${OUT_DIR}_transe"
# HUGG_OUT_DIR_TRanse="${HUGG_OUT_DIR}_transe"
# for BASE_MODEL in "${MASKED_MODELS[@]}"; do
#         # 1. Assign Hyperparameters based on the model name
#         case "$BASE_MODEL" in
#             # "nlpaueb/bert-base-uncased-eurlex")    NSP_LAMBDA=0.4; CLS_LAMBDA=0.4 ;;
#             # "nlpaueb/bert-base-uncased-contracts") NSP_LAMBDA=0.3; CLS_LAMBDA=0.7 ;;
#             "nlpaueb/bert-base-uncased-echr")      NSP_LAMBDA=0.3; CLS_LAMBDA=0.4 ;;
#             # "casehold/custom-legalbert")           NSP_LAMBDA=0.4; CLS_LAMBDA=0.5 ;;
#             # "dlicari/Italian-Legal-BERT")          NSP_LAMBDA=0.5; CLS_LAMBDA=0.1 ;;
#             # "google-bert/bert-base-uncased")       NSP_LAMBDA=0.3; CLS_LAMBDA=0.6 ;;
#             # "nlpaueb/legal-bert-base-uncased")     NSP_LAMBDA=0.2; CLS_LAMBDA=0.2 ;;
#             # "nlpaueb/legal-bert-small-uncased")    NSP_LAMBDA=0.6; CLS_LAMBDA=0.1 ;;
#             # "avichr/Legal-heBERT")                 NSP_LAMBDA=0.0; CLS_LAMBDA=1.0 ;;
#             *) echo "No specific hyperparameters for $BASE_MODEL, using defaults." ;;
#         esac
#         CUDA_VISIBLE_DEVICES=${GPU_ID} python ../finetune_multi_task_STvO_mlm_clm_clu.py \
#             --translator_model_name "m2m100_418M" \
#             --clm_data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph.dot \
#             --nsp_data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph_nsp_dataset.json \
#             --clu_data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph_clu_dataset.json \
#             --cls_data_path ../data/synthetic-data/stvo/qa_train.jsonl \
#             --paragraph_embeddings_file ../legal-KGE/embeddings/transe/STvO/transe_paragraph_embeddings_best.json \
#             --train_history_path "training_history_job$SLURM_JOB_ID.json" \
#             --plot_path "training_progress_job$SLURM_JOB_ID.png" \
#             --wandb_project "transe-LLM-Drive-License-structured" \
#             --base_model_name $BASE_MODEL \
#             --modeling_type 'masked' \
#             --hf_cache_dir $HF_DIR \
#             --output_dir $out_DIR_TRanse \
#             --hugg_out_dir $HUGG_OUT_DIR_TRanse \
#             --num_epochs 20\
#             --batch_size_clu 16\
#             --batch_size_nsp 16\
#             --batch_size_cls 16\
#             --learning_rate 5e-5\
#             --clm_nsp_lambdas $NSP_LAMBDA\
#             --cls_lambdas $CLS_LAMBDA\
#             --gradient_accumulation_steps 2\
#             --clm_block_size 512\
#             --max_seq_length_clu 128\
#             --max_seq_length_nsp 128 \
#             --max_seq_length_cls 128 \
#             --max_segments 10\
#             --seed 42 >> "$LOG_OUT" 2>> "$LOG_ERR"
# done





# repeat the experiment with rotate, complex, and distmult embeddings.
# rotate
# add rotate to outdir
out_DIR_ROTATE="${OUT_DIR}_rotate"
HUGG_OUT_DIR_ROTATE="${HUGG_OUT_DIR}_rotate"
for BASE_MODEL in "${MASKED_MODELS[@]}"; do
       # 1. Assign Hyperparameters based on the model name
        case "$BASE_MODEL" in
            "nlpaueb/bert-base-uncased-echr")      NSP_LAMBDA=0.3; CLS_LAMBDA=0.4 ;;
            *) echo "No specific hyperparameters for $BASE_MODEL, using defaults." ;;
        esac
        CUDA_VISIBLE_DEVICES=${GPU_ID}
        python ../finetune_multi_task_STvO_mlm_clm_clu.py \
            --translator_model_name "m2m100_418M" \
            --clm_data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph.dot \
            --nsp_data_path ../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung_graph_nsp_dataset.json \
            --clu_data_path ../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung_graph_sentence_clu_dataset.json \
            --cls_data_path ../data/synthetic-data/stvo/qa_train.jsonl \
            --paragraph_embeddings_file ../legal-KGE/embeddings/rotate/m2m100_418M_STvO/rotate_paragraph_embeddings_best.json \
            --train_history_path "training_history_job$SLURM_JOB_ID.json" \
            --plot_path "training_progress_job$SLURM_JOB_ID.png" \
            --wandb_project "rotate-LLM-Drive-License-structured-Understanding" \
            --base_model_name $BASE_MODEL \
            --modeling_type 'masked' \
            --hf_cache_dir $HF_DIR \
            --output_dir $out_DIR_ROTATE \
            --hugg_out_dir $HUGG_OUT_DIR_ROTATE \
            --num_epochs 20\
            --batch_size_clu 16\
            --batch_size_nsp 16\
            --batch_size_cls 16\
            --learning_rate 5e-5\
            --clm_nsp_lambdas 0.2\
            --cls_lambdas 0.5\
            --gradient_accumulation_steps 2\
            --clm_block_size 512\
            --max_seq_length_clu 128\
            --max_seq_length_nsp 128 \
            --max_seq_length_cls 128 \
            --max_segments 5\
            --seed 42 >> "$LOG_OUT" 2>> "$LOG_ERR"

done

# complex
out_DIR_COMPLEX="${OUT_DIR}_complex"
HUGG_OUT_DIR_COMPLEX="${HUGG_OUT_DIR}_complex"
for BASE_MODEL in "${MASKED_MODELS[@]}"; do
        # 1. Assign Hyperparameters based on the model name
        case "$BASE_MODEL" in
            "nlpaueb/bert-base-uncased-echr")      NSP_LAMBDA=0.3; CLS_LAMBDA=0.4 ;;
            *) echo "No specific hyperparameters for $BASE_MODEL, using defaults." ;;
        esac
        CUDA_VISIBLE_DEVICES=${GPU_ID}
        python ../finetune_multi_task_STvO_mlm_clm_clu.py \
            --translator_model_name "m2m100_418M" \
            --clm_data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph.dot \
            --nsp_data_path ../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung_graph_nsp_dataset.json \
            --clu_data_path ../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung_graph_sentence_clu_dataset.json \
            --cls_data_path ../data/synthetic-data/stvo/qa_train.jsonl \
            --paragraph_embeddings_file ../legal-KGE/embeddings/complex/m2m100_418M_STvO/complex_paragraph_embeddings_best.json \
            --train_history_path "training_history_job$SLURM_JOB_ID.json" \
            --plot_path "training_progress_job$SLURM_JOB_ID.png" \
            --wandb_project "complex-LLM-Drive-License-structured-Understanding" \
            --base_model_name $BASE_MODEL \
            --modeling_type 'masked' \
            --hf_cache_dir $HF_DIR \
            --output_dir $out_DIR_COMPLEX \
            --hugg_out_dir $HUGG_OUT_DIR_COMPLEX \
            --num_epochs 20\
            --batch_size_clu 16\
            --batch_size_nsp 16\
            --batch_size_cls 16\
            --learning_rate 5e-5\
            --clm_nsp_lambdas 0.2\
            --cls_lambdas 0.5\
            --gradient_accumulation_steps 2\
            --clm_block_size 512\
            --max_seq_length_clu 128\
            --max_seq_length_nsp 128 \
            --max_seq_length_cls 128 \
            --max_segments 5\
            --seed 42 >> "$LOG_OUT" 2>> "$LOG_ERR"
done


# distmult
out_DIR_DISTMULT="${OUT_DIR}_distmult"
HUGG_OUT_DIR_DISTMULT="${HUGG_OUT_DIR}_distmult"
for BASE_MODEL in "${MASKED_MODELS[@]}"; do
        # 1. Assign Hyperparameters based on the model name
        case "$BASE_MODEL" in
            "nlpaueb/bert-base-uncased-echr")      NSP_LAMBDA=0.3; CLS_LAMBDA=0.4 ;;
            *) echo "No specific hyperparameters for $BASE_MODEL, using defaults." ;;
        esac
        CUDA_VISIBLE_DEVICES=${GPU_ID}
        python ../finetune_multi_task_STvO_mlm_clm_clu.py \
            --translator_model_name "m2m100_418M" \
            --clm_data_path ../data/stvo/m2m100_418M_translated_integrated_Straßenverkehrs_Ordnung_graph.dot \
            --nsp_data_path ../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung_graph_nsp_dataset.json \
            --clu_data_path ../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung_graph_sentence_clu_dataset.json \
            --cls_data_path ../data/synthetic-data/stvo/qa_train.jsonl \
            --paragraph_embeddings_file ../legal-KGE/embeddings/distmult/m2m100_418M_STvO/distmult_paragraph_embeddings_best.json \
            --train_history_path "training_history_job$SLURM_JOB_ID.json" \
            --plot_path "training_progress_job$SLURM_JOB_ID.png" \
            --wandb_project "distmult-LLM-Drive-License-structured-Understanding" \
            --base_model_name $BASE_MODEL \
            --modeling_type 'masked' \
            --hf_cache_dir $HF_DIR \
            --output_dir $out_DIR_DISTMULT \
            --hugg_out_dir $HUGG_OUT_DIR_DISTMULT \
            --num_epochs 20\
            --batch_size_clu 16\
            --batch_size_nsp 16\
            --batch_size_cls 16\
            --learning_rate 5e-5\
            --clm_nsp_lambdas 0.2\
            --cls_lambdas 0.5\
            --gradient_accumulation_steps 2\
            --clm_block_size 512\
            --max_seq_length_clu 128\
            --max_seq_length_nsp 128 \
            --max_seq_length_cls 128 \
            --max_segments 5\
            --seed 42 >> "$LOG_OUT" 2>> "$LOG_ERR"
done



# # Experimental conditions:
# experiments = {
#     # Control groups
#     'baseline_no_kg': {
#         'tasks': ['CLM only'],  # No KG integration
#         'expected': 'Lower performance'
#     },
    
#     # Your current approach
#     'current_kg_clustering': {
#         'tasks': ['CLM + CLU (cosine embedding)'],
#         'expected': 'Modest improvement'
#     },
    
#     # Improved KG integration strategies
#     'explicit_kg_attention': {
#         'tasks': ['CLM + KG-aware attention'],
#         'expected': 'Significant improvement'
#     },
    
#     'hierarchical_kg_injection': {
#         'tasks': ['CLM + Hierarchical KG routing'],
#         'expected': 'Best performance'
#     }
# }