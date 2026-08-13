# Paths and parameters
ROOT_DIR="/temp/siddig"
DATA_DIR="$ROOT_DIR/driving-license"
HF_DIR="$ROOT_DIR/driving-license/HF_models"
FINE_TUNED_MODELS_DIR="$ROOT_DIR/driving-license/STVo_PRLM_HF_hierarchical_models_20_epochs_10_segments"

# Configuration for the Tmux Session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_OUT="logs/out/tmux-STVO-train-$TIMESTAMP.out"
LOG_ERR="logs/err/tmux-STVO-train-$TIMESTAMP.err"

# Ensure log directories exist
mkdir -p logs/out logs/err 
mkdir -p $DATA_DIR 

# Create and activate the virtual environment if it doesn't already exist
if [ ! -d "$DATA_DIR/envs/driving-eval-env" ]; then
    python3 -m venv $DATA_DIR/envs/driving-eval-env
    source $DATA_DIR/envs/driving-eval-env/bin/activate
    bash ./install.sh
fi 


source $DATA_DIR/envs/driving-eval-env/bin/activate
# bash ./install.sh

GPU_ID=3  # Change this to the appropriate GPU ID if you have multiple GPUs
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Verifying GPU visibility for assigned ID: $CUDA_VISIBLE_DEVICES"
$PYTHON_EXE -c "import torch; print(f'Script sees CUDA: {torch.cuda.is_available()}')"

# Complete Pipeline for Citation-based Legal Q&A Dataset Generation and Evaluation
# ================================================================================

echo "=========================================="
echo "Citation-based Legal Q&A Dataset Pipeline"
echo "=========================================="



# Configuration
# MODEL_NAME="bert-base-uncased"
HF_TOKEN="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"  # Replace with actual token
CUURENT_DATE_TIME=$(date +"%Y%m%d_%H%M%S")
EVALUATION_DIR="./results/kg_ablation/evaluation_$CUURENT_DATE_TIME"
cache_dir="$DATA_DIR/HF_models/"
access_token="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"
ARGS_OUT="$DATA_DIR/LLM_evaluator"



# python combine_and_clean_synthetic_QA.py \
#     --input ../synthetic-data/stvo/qa_chocolatine_balanced.jsonl \
#     ../synthetic-data/stvo/qa_mistral_fast.jsonl \
#     ../synthetic-data/stvo/qa_llama_highquality.jsonl \
#     ../synthetic-data/stvo/qa_domain_contract.jsonl \
#     ../synthetic-data/stvo/qa_domain_regulatory.jsonl \
#     ../synthetic-data/stvo/qa_domain_traffic_law.jsonl \
#     ../synthetic-data/stvo/qa_threshold_0.55_mistral.jsonl \
#     ../synthetic-data/stvo/qa_threshold_0.60_mistral.jsonl \
#     ../synthetic-data/stvo/qa_threshold_0.65_mistral.jsonl \
#     ../synthetic-data/stvo/qa_threshold_0.70_mistral.jsonl \
#     ../synthetic-data/stvo/qa_llama_incremental_total.jsonl \
#     --output $DATA_DIR/LLM_evaluator/qa.jsonl \
#     --stats $DATA_DIR/LLM_evaluator/qa_stats.json


DATASET_FILE="../synthetic-data/stvo/qa_test.jsonl"
# MASKED_MODELS=(
#     # Best performing models 
#     #TOP
#     nlpaueb/bert-base-uncased-eurlex
#     bert-base-uncased-eurlex_m2m100_418M_clm_nsp_0.4_clu_0.2_cls_0.4
#     bert-base-uncased-eurlex_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5/
#     nlpaueb/bert-base-uncased-contracts
#     bert-base-uncased-contracts_m2m100_418M_clm_nsp_0.3_clu_0.0_cls_0.7
#     bert-base-uncased-contracts_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5/
#     nlpaueb/bert-base-uncased-echr
#     bert-base-uncased-echr_m2m100_418M_clm_nsp_0.3_clu_0.3_cls_0.4
#     bert-base-uncased-echr_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5/

#     casehold/custom-legalbert
#     custom-legalbert_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5
    

#     dlicari/Italian-Legal-BERT
#     Italian-Legal-BERT_m2m100_418M_clm_nsp_0.5_clu_0.4_cls_0.1
#     Italian-Legal-BERT_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5/

#     # Moderate
#     google-bert/bert-base-uncased
#     bert-base-uncased_m2m100_418M_clm_nsp_0.3_clu_0.1_cls_0.6
#     nlpaueb/legal-bert-base-uncased 
#     legal-bert-base-uncased_m2m100_418M_clm_nsp_0.2_clu_0.6_cls_0.2

#     # SMALL
#     nlpaueb/legal-bert-small-uncased
#     legal-bert-small-uncased_m2m100_418M_clm_nsp_0.6_clu_0.3_cls_0.1
#     avichr/Legal-heBERT
#     Legal-heBERT_m2m100_418M_clm_nsp_0.0_clu_0.0_cls_1.0
    

#     # German trained models.
#     # nlpaueb/bert-base-uncased-eurlex
#     # bert-base-uncased-eurlex_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5/
#     # nlpaueb/bert-base-uncased-contracts
#     # bert-base-uncased-contracts_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5/
#     # nlpaueb/bert-base-uncased-echr
#     # bert-base-uncased-echr_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5/
#     # casehold/custom-legalbert
#     # custom-legalbert_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5/
#     # dlicari/Italian-Legal-BERT
#     # Italian-Legal-BERT_m2m100_418M_clm_nsp_0.4_clu_0.1_cls_0.5/
# )

BASE_MODELS=(
    # nlpaueb/bert-base-uncased-eurlex
    # nlpaueb/bert-base-uncased-contracts
    nlpaueb/bert-base-uncased-echr
    # casehold/custom-legalbert
    # dlicari/Italian-Legal-BERT
    # google-bert/bert-base-uncased
    # nlpaueb/legal-bert-base-uncased 
    # nlpaueb/legal-bert-small-uncased
    # avichr/Legal-heBERT
)


KGS=(
  rotate
  distmult
  complex
  transe
)

for KG in "${KGS[@]}"; do

FIN_DIR="${FINE_TUNED_MODELS_DIR}_$KG"
EVAL_DIR="${EVALUATION_DIR}_20epochs_10segs_$KG"
mkdir -p $EVAL_DIR/figures

# echo "Auto-discovering models in: $FINE_TUNED_MODELS_DIR"
MULTITASK_MODELS=()
shopt -s nullglob
for MODEL_DIR in "$FIN_DIR"/*; do
    if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
        MODEL_NAME=$(basename "$MODEL_DIR")
        MULTITASK_MODELS+=("$MODEL_NAME")
    fi
done
shopt -u nullglob

# --- 2. Organize and Deduplicate ---
MASKED_MODELS=()
# We keep track of which tuned models we've already "used"
USED_TUNED_MODELS=()

for BASE_MODEL in "${BASE_MODELS[@]}"; do
    BASE_MODEL_NAME=$(basename "$BASE_MODEL")
    
    # Always add the Base Model first
    MASKED_MODELS+=("$BASE_MODEL")
    
    for TUNED_MODEL in "${MULTITASK_MODELS[@]}"; do
        # Use a Prefix Match (starts with) instead of Substring Match (contains)
        # This prevents 'legal-bert-base' from matching 'bert-base'
        if [[ "$TUNED_MODEL" == "${BASE_MODEL_NAME}"_* ]]; then
            
            # Additional check: Has this specific tuned model been added already?
            if [[ ! " ${USED_TUNED_MODELS[@]} " =~ " ${TUNED_MODEL} " ]]; then
                MASKED_MODELS+=("$TUNED_MODEL")
                USED_TUNED_MODELS+=("$TUNED_MODEL")
                echo "Mapped: $BASE_MODEL_NAME -> $TUNED_MODEL"
            fi
        fi
    done
done

echo "Total entities in MASKED_MODELS (Base + Unique Tuned): ${#MASKED_MODELS[@]}"

# Print the final list of models to be evaluated
echo "Final list of models to be evaluated:"
for MODEL in "${MASKED_MODELS[@]}"; do
    echo " - $MODEL"
done

# number of models to be evaluated
echo "Total models to be evaluated: ${#MASKED_MODELS[@]}"



# Step 1: Run Dataset Analysis
echo "Step 1: Running dataset analysis..."
    python evaluation_suite.py \
    --dataset_path "$DATASET_FILE" \
    --output_dir "$EVAL_DIR" \
    --run_analysis >> "$LOG_OUT" 2>> "$LOG_ERR"

# STEP 2: Run Model Evaluation
echo "Step 2: ZERO-SHOT EVALUATION for masked models..."
for MODEL_NAME in "${MASKED_MODELS[@]}"; do
    echo "Evaluating model: $MODEL_NAME"
        python evaluation_suite.py \
        --dataset_path $DATASET_FILE \
        --base_model $MODEL_NAME \
        --output_dir $EVAL_DIR \
        --training_args_out $ARGS_OUT \
        --fine_tuned_dir $FIN_DIR \
        --run_fine_tuning \
        --cache_dir $cache_dir \
        --access_token $access_token \
        --binary_classification \
        --zero_shot_eval >> "$LOG_OUT" 2>> "$LOG_ERR"
done 

# # Step 3: Run Fine-tuning and Evaluation for Masked Models
# echo "Step 3: Running fine-tuning and evaluation for masked models..."
# for MODEL_NAME in "${MASKED_MODELS[@]}"; do
#     python evaluation_suite.py \
#         --dataset_path $DATASET_FILE \
#         --base_model $MODEL_NAME \
#         --output_dir $EVAL_DIR \
#         --training_args_out $ARGS_OUT \
#         --fine_tuned_dir $FIN_DIR \
#         --run_fine_tuning \
#         --cache_dir $cache_dir \
#         --access_token $access_token \
#         --binary_classification \
#         --epochs 5\
#         --train_size 0.6\
#         --val_size 0.1\
#         --test_size 0.3 >> "$LOG_OUT" 2>> "$LOG_ERR"

#         echo "Finished evaluating base model: $MODEL_NAME"
# done
done



# # Step 2: Run Model Evaluation
# echo "Step 2: Running model evaluation and baselines..."
# python evaluation_suite.py \
#     --dataset_path "$DATASET_FILE" \
#     --output_dir "$EVALUATION_DIR" \
#     --run_evaluation \
