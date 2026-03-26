#!/bin/bash

# Configuration for the Tmux Session
SESSION_NAME="stvo_eval_session"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_OUT="out/tmux-STVO-eval-$TIMESTAMP.out"
LOG_ERR="err/tmux-STVO-eval-$TIMESTAMP.err"

# Ensure log directories exist
mkdir -p out err

# --- PATHS AND PARAMETERS ---
ROOT_DIR="/temp/siddig"
DATA_DIR="$ROOT_DIR/driving-license"
HUGG_OUT_DIR="$DATA_DIR/STVo_PRLM_HF_hierarchical_models_20_epochs_10_segments"
OUT_DIR="$DATA_DIR/STVo_PRLM_PY_hierarchical_models_20_epochs_10_segments"
HF_DIR="$ROOT_DIR/HF_models"

mkdir -p $DATA_DIR 

# --- GPU SETUP ---
# Since we are not using Slurm, we manually specify the GPU ID (0-based)
# Use 'nvidia-smi' to find available indices.
export CUDA_VISIBLE_DEVICES=1

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


# =====================================================================================
MODEL_CHECKPOINT_DIR="$DATA_DIR/STVo_PRLM_PY_hierarchical_models_20_epochs_10_segments_transe"  
# /fast_storage/siddig/driving-license/STVo_PRLM_PY_hierarchical_models_test_20_epochs_5_segments_all/
mkdir -p $MODEL_CHECKPOINT_DIR



MASKED_MODELS=(

    nlpaueb/bert-base-uncased-eurlex
    # bert-base-uncased-eurlex_m2m100_418M_clm_nsp_0.4_clu_0.2_cls_0.4
    nlpaueb/bert-base-uncased-contracts
    # bert-base-uncased-contracts_m2m100_418M_clm_nsp_0.3_clu_0.0_cls_0.7
    nlpaueb/bert-base-uncased-echr
    # bert-base-uncased-echr_m2m100_418M_clm_nsp_0.3_clu_0.3_cls_0.4
    casehold/custom-legalbert
    # custom-legalbert_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5
    dlicari/Italian-Legal-BERT
    # Italian-Legal-BERT_m2m100_418M_clm_nsp_0.5_clu_0.4_cls_0.1
    google-bert bert-base-uncased
    # bert-base-uncased_m2m100_418M_clm_nsp_0.3_clu_0.1_cls_0.6
    nlpaueb/legal-bert-base-uncased
    # legal-bert-base-uncased_m2m100_418M_clm_nsp_0.2_clu_0.6_cls_0.2
    nlpaueb/legal-bert-small-uncased
    # legal-bert-small-uncased_m2m100_418M_clm_nsp_0.6_clu_0.3_cls_0.1
    avichr/Legal-heBERT
    # Legal-heBERT_m2m100_418M_clm_nsp_0.0_clu_0.0_cls_1.0
)


# CAUSAL_MODELS=(
#     Qwen/Qwen3-Embedding-0.6B
#     google/gemma-3-270m
#     google/gemma-3-270m-it
#     mistralai/Mistral-7B-v0.1
#     Qwen/Qwen-7B
#     jpacifico/Chocolatine-14B-Instruct-DPO-v1.2
#     # openai-community/gpt2

#     # ibm-granite/granite-3.1-1b-a400m-instruct
#     # Gensyn/Qwen2.5-0.5B-Instruct
#     # Qwen/Qwen3-Embedding-0.6B
#     # meta-llama/Llama-3.2-1B
#     # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#     # Qwen/Qwen2.5-Math-1.5B

# )

CLM_LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
CLS_LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

TRANSLATORS=(
    m2m100_418M
    # opus-mt-de-en
)

SIMILARITY_THRESHOLD_VALUES=(0.5 0.6 0.7 0.8 0.9)
SIMILARITY_THRESHOLD=${SIMILARITY_THRESHOLD_VALUES[1]}

# Iterate Over MASKED MODELS
for BASE_MODEL_NAME in "${MASKED_MODELS[@]}"; do
    
    echo "==================================================================="
    echo "--- 1.EVALUATING BASELINE: ${BASE_MODEL_NAME} ---"
    echo "==================================================================="
    

    python ../eval.py --data_path "../data/test-sets/austrian-driving-license/fragen"\
                --paragraph_embeddings_file "../legal-KGE/embeddings/transe/m2m100_418M_STvO/transe_paragraph_embeddings_best.json"\
                --data_name "austrian"\
                --baseline_model $BASE_MODEL_NAME\
                --seed 42\
                --decoding "combine"\
                --batch_samples 20\
                --similarity_threshold $SIMILARITY_THRESHOLD\
                --skip_images \
                --skip_category "direct_answer"\
                --eval_base \
                --modeling_type "masked"\
                --task_type "cls"\
                --num_cls_labels 2 \
                --max_seq_length_clm 128\
                --max_seq_length_clu 128\
                --max_seq_length_cls 128\
                --max_seq_length_nsp 128\
                --max_num_segments 20\
                --max_seq_length 2048\
                --max_new_tokens 2\
                --top_p 1\
                --top_k 40\
                --temperature 0.2\
                --prompt_version 555\
                --temperatures 0.2\
                --repetition_penalty 1.2\
                --skip_images \
                --results_file "../results/model-performance-clm_nsp_clu_cls-austrian-$SLURM_JOB_ID-stvo-20_epoch_10_segments_transe.csv"\
                --cache_dir $HF_DIR  >> $LOG_OUT 2>> $LOG_ERR

    echo "==================================================================="
    echo "--- Finished EVALUATING BASELINE: ${BASE_MODEL_NAME} ---"
    echo "==================================================================="

    # Iterate over each combination of lambda weights
    for CLM_LAMBDA in "${CLM_LAMBDAS[@]}"; do
        for CLS_LAMBDA in "${CLS_LAMBDAS[@]}"; do
            # Iterate over each translator model
            for TRANSLATOR in "${TRANSLATORS[@]}"; do
                # Skip if the sum of CLM and clu lambda is greater than or equal to 1.0
                # Calculate the sum of CLM and clu lambdas
                
                # calculate triplet the format is 1 - (CLM + clu) with 2 decimal places
                CLU_LAMBDA=$(echo "scale=1; 1 - ($CLM_LAMBDA + $CLS_LAMBDA)" | bc)
                # CLU_LAMBDA=1.0

                if (( $(echo "$CLU_LAMBDA < 0" | bc -l) )); then
                    echo "Skipping combination: CLM=${CLM_LAMBDA}, cls=${CLS_LAMBDA}, clu would be negative."
                    continue
                fi
                
                # make sure the value is approximated to .2 decimal places
                CLM_LAMBDA=$(printf "%.1f" $CLM_LAMBDA)
                CLU_LAMBDA=$(printf "%.1f" $CLU_LAMBDA)
                CLS_LAMBDA=$(printf "%.1f" $CLS_LAMBDA)


                MODEL_CHECKPOINT="${TRANSLATOR}_clm_nsp_${CLM_LAMBDA}_clu_${CLU_LAMBDA}_cls_${CLS_LAMBDA}.pt"
                echo "-------------------------------------------------------------------"
                echo "--- Starting EVALUATING TUNED MODEL: ${MODEL_CHECKPOINT} ---"
                echo "--- Base Model: ${BASE_MODEL_NAME}"
                echo "--- Lambdas: CLM=${CLM_LAMBDA}, clu=${CLU_LAMBDA}, cls=${CLS_LAMBDA}"
                echo "-------------------------------------------------------------------"

                python ../eval.py --data_path "../data/test-sets/austrian-driving-license/fragen"\
                            --paragraph_embeddings_file "../legal-KGE/embeddings/transe/${TRANSLATOR}_STvO/transe_paragraph_embeddings_best.json"\
                            --data_name "austrian"\
                            --baseline_model $BASE_MODEL_NAME \
                            --model_checkpoint_dir $MODEL_CHECKPOINT_DIR\
                            --model_checkpoint $MODEL_CHECKPOINT\
                            --seed 42\
                            --decoding "combine"\
                            --batch_samples 20\
                            --similarity_threshold $SIMILARITY_THRESHOLD\
                            --skip_images \
                            --modeling_type "masked"\
                            --task_type "cls"\
                            --num_cls_labels 2 \
                            --max_seq_length_clm 128\
                            --max_seq_length_clu 128\
                            --max_seq_length_cls 128\
                            --max_seq_length_nsp 128\
                            --max_num_segments 20\
                            --max_seq_length 2048\
                            --max_new_tokens 2\
                            --top_p 1\
                            --top_k 40\
                            --temperature 0.2\
                            --prompt_version 555\
                            --temperatures 0.2\
                            --repetition_penalty 1.1\
                            --skip_images \
                            --results_file "../results/model-performance-clm_nsp_clu_cls-austrian-$SLURM_JOB_ID-stvo-20_epoch_10_segments_transe.csv"\
                            --cache_dir $HF_DIR  >> $LOG_OUT 2>> $LOG_ERR

                echo "-------------------------------------------------------------------"
                echo "--- Finished EVALUATING TUNED MODEL: ${MODEL_CHECKPOINT} ---"
                echo "-------------------------------------------------------------------"
            done
    done
done
done    


# #################################################  CAUSAL MODELS  ##########################################################



# # Iterate Over CAUSAL MODELS
# for BASE_MODEL_NAME in "${CAUSAL_MODELS[@]}"; do
#     echo "==================================================================="
#     echo "--- 1.EVALUATING BASELINE: ${BASE_MODEL_NAME} ---"
#     echo "==================================================================="

#     python ../eval.py --data_path "../data/test-sets/austrian-driving-license/fragen"\
#                 --paragraph_embeddings_file "../transE_embeddings/m2m100_418M_STvO/paragraph_embeddings_lists.json"\
#                 --data_name "austrian"\
#                 --baseline_model $BASE_MODEL_NAME\
#                 --seed 42\
#                 --decoding "combine"\
#                 --batch_samples 20\
#                 --similarity_threshold $SIMILARITY_THRESHOLD\
#                 --skip_images \
#                 --skip_category "direct_answer"\
#                 --eval_base \
#                 --modeling_type "causal"\
#                 --task_type "cls"\
#                 --num_cls_labels 2 \
#                 --max_seq_length_clm 128\
#                 --max_seq_length_clu 128\
#                 --max_seq_length_cls 128\
#                 --max_num_segments 20\
#                 --max_seq_length 2048\
#                 --max_new_tokens 2\
#                 --top_p 1\
#                 --top_k 40\
#                 --temperature 0.2\
#                 --prompt_version 555\
#                 --temperatures 0.2\
#                 --repetition_penalty 1.2\
#                 --skip_images \
#                 --results_file "../results/model-performance-clm_mlm_clu_cls-austrian-$SLURM_JOB_ID-stvo-20_epoch_5_segments_cls.csv"\
#                 --cache_dir $HF_DIR \

# echo "==================================================================="
# echo "--- Finished EVALUATING BASELINE: ${BASE_MODEL_NAME} ---"
# echo "==================================================================="

# # Iterate over each combination of lambda weights
# for CLM_LAMBDA in "${CLM_LAMBDAS[@]}"; do
#     for CLS_LAMBDA in "${CLS_LAMBDAS[@]}"; do
#         # Iterate over each translator model
#         for TRANSLATOR in "${TRANSLATORS[@]}"; do
#             # Skip if the sum of CLM and clu lambda is greater than or equal to 1.0
#             # Calculate the sum of CLM and clu lambdas
            
#             # calculate triplet the format is 1 - (CLM + clu) with 2 decimal places
#             CLU_LAMBDA=$(echo "scale=1; 1 - ($CLM_LAMBDA + $CLS_LAMBDA)" | bc)
#             if (( $(echo "$CLU_LAMBDA < 0" | bc -l) )); then
#                 echo "Skipping combination: CLM=${CLM_LAMBDA}, cls=${CLS_LAMBDA}, clu would be negative."
#                 continue
#             fi
#             # make sure the value is approximated to .2 decimal places
#             CLM_LAMBDA=$(printf "%.1f" $CLM_LAMBDA)
#             CLU_LAMBDA=$(printf "%.1f" $CLU_LAMBDA)
#             CLS_LAMBDA=$(printf "%.1f" $CLS_LAMBDA)


#             MODEL_CHECKPOINT="${TRANSLATOR}_clm_mlm_${CLM_LAMBDA}_clu_${CLU_LAMBDA}.pt"
#             echo "-------------------------------------------------------------------"
#             echo "--- Starting EVALUATING TUNED MODEL: ${MODEL_CHECKPOINT} ---"
#             echo "--- Base Model: ${BASE_MODEL_NAME}"
#             echo "--- Lambdas: CLM=${CLM_LAMBDA}, clu=${CLU_LAMBDA}, cls=${CLS_LAMBDA}"
#             echo "-------------------------------------------------------------------"

#             python ../eval.py --data_path "../data/test-sets/austrian-driving-license/fragen"\
#                         --paragraph_embeddings_file "../transE_embeddings/${TRANSLATOR}_STvO/paragraph_embeddings_lists.json"\
#                         --data_name "austrian"\
#                         --baseline_model $BASE_MODEL_NAME \
#                         --model_checkpoint_dir $MODEL_CHECKPOINT_DIR\
#                         --model_checkpoint $MODEL_CHECKPOINT\
#                         --seed 42\
#                         --decoding "combine"\
#                         --batch_samples 20\
#                         --similarity_threshold $SIMILARITY_THRESHOLD\
#                         --skip_images \
#                         --modeling_type "causal"\
#                         --task_type "cls"\
#                         --num_cls_labels 2 \
#                         --max_seq_length_clm 128\
#                         --max_seq_length_clu 128\
#                         --max_seq_length_cls 128\
#                         --max_num_segments 20\
#                         --max_seq_length 2048\
#                         --max_new_tokens 2\
#                         --top_p 1\
#                         --top_k 40\
#                         --temperature 0.2\
#                         --prompt_version 555\
#                         --temperatures 0.2\
#                         --repetition_penalty 1.1\
#                         --skip_images \
#                         --results_file "../results/model-performance-clm_mlm_clu_cls-austrian-$SLURM_JOB_ID-stvo-20_epoch_5_segments_cls.csv"\
#                         --cache_dir $HF_DIR \

#             echo "-------------------------------------------------------------------"
#             echo "--- Finished EVALUATING TUNED MODEL: ${MODEL_CHECKPOINT} ---"
#             echo "-------------------------------------------------------------------"
#         done
#     done
# done
# done    
