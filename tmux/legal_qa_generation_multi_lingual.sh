# Save echo device info into CSV file
mkdir -p ./MIG-IDs
echo $SLURM_JOB_ID, $CUDA_VISIBLE_DEVICES > ./MIG-IDs/advance-monitor-JOBID-$SLURM_JOB_ID.csv
nvidia-smi -L >> ./MIG-IDs/advance-monitor-JOBID-$SLURM_JOB_ID.csv
nvidia-smi >> ./MIG-IDs/advance-monitor-JOBID-$SLURM_JOB_ID.csv



# Paths and parameters
ROOT_DIR="/temp/siddig/"
DATA_DIR="$ROOT_DIR/driving-license"
OUT_DIR="$ROOT_DIR/driving-license/multitask_models"
HF_DIR="$ROOT_DIR/driving-license/HF_models"

mkdir -p $DATA_DIR

# Create and activate the virtual environment if it doesn't already exist
if [ ! -d "$DATA_DIR/envs/driving-env" ]; then
    python3 -m venv $DATA_DIR/envs/driving-env
    source $DATA_DIR/envs/driving-env/bin/activate
    bash ./install.sh
fi 

source $DATA_DIR/envs/driving-env/bin/activate
# bash ./install.sh
     

# Legal Q&A Generation: Recommended Execution Examples


# # ============================================================================
# # SCENARIO 0: High-Quality Generation with Llama for German (BASELINE)
# # ============================================================================

# echo "=== SCENARIO 0.1: fast Generation with Mistral for German ==="
python ../legal_qa_generation_multi_lingual.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
    --num_samples 5000 \
    --pairs_per_sentence 4 \
    --domain_type "traffic_law" \
    --language "de" \
    --min_quality_score 0.60 \
    --max_retries 4\
    --shuffle_sentences \
    --hf_token "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" \
    --input_file "../data/stvo/parsed_main_content_Straßenverkehrs_Ordnung_de.json" \
    --output_file "../data/synthetic-data/stvo/de/qa_mistral_fast_de.jsonl"

# echo "=== SCENARIO 0.2: High-Quality Generation with Llama for German ==="
# python ../legal_qa_generation_multi_lingual.py \
#     --model_name "meta-llama/Llama-2-7b-chat-hf" \
#     --num_samples 5000 \
#     --pairs_per_sentence 4 \
#     --domain_type "traffic_law" \
#     --language "de" \
#     --min_quality_score 0.65 \
#     --max_retries 3 \
#     --shuffle_sentences \
#     --hf_token "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" \
#     --input_file "../data/stvo/parsed_main_content_Straßenverkehrs_Ordnung_de.json" \
#     --output_file "../data/synthetic-data/stvo/qa_llama_highquality_de.jsonl"

# echo "=== SCENARIO 0.3: Balanced Generation with jpacifico/Chocolatine for German ==="
# python ../legal_qa_generation_multi_lingual.py \
#     --model_name "jpacifico/Chocolatine-14B-Instruct-DPO-v1.2" \
#     --num_samples 5000 \
#     --pairs_per_sentence 4 \
#     --domain_type "traffic_law" \
#     --language "de" \
#     --min_quality_score 0.65 \
#     --max_retries 4\
#     --shuffle_sentences \
#     --hf_token "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" \
#     --input_file "../data/stvo/parsed_main_content_Straßenverkehrs_Ordnung_de.json" \
#     --output_file "../data/synthetic-data/stvo/qa_chocolatine_balanced_de.jsonl"



# # ============================================================================
# # SCENARIO 1: High-Quality Generation with Llama (RECOMMENDED)
# # ============================================================================
# # Best for: Production datasets requiring high accuracy
# # Expected: Quality scores 0.70-0.85, slower but reliable
# # Time: ~2-4 hours for 5000 samples on RTX 3090

# echo "=== SCENARIO 1: High-Quality Llama Generation ==="

# python ../legal_qa_generation_multi_lingual.py \
#     --model_name "meta-llama/Llama-2-7b-chat-hf" \
#     --num_samples 5000 \
#     --pairs_per_sentence 4 \
#     --domain_type "traffic_law" \
#     --language "en" \
#     --min_quality_score 0.65 \
#     --max_retries 3 \
#     --shuffle_sentences \
#     --hf_token "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" \
#     --input_file "../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung.json" \
#     --output_file "../data/synthetic-data/stvo/qa_llama_highquality.jsonl"


# # ============================================================================
# # SCENARIO 2: Fast Generation with Mistral (CURRENT SETUP)
# # ============================================================================
# # Best for: Quick prototyping, testing, baseline generation
# # Expected: Quality scores 0.60-0.72, faster inference
# # Time: ~1-2 hours for 5000 samples on RTX 3090

# echo "=== SCENARIO 2: Fast Generation with Mistral ==="

# python ../legal_qa_generation_multi_lingual.py \
#     --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
#     --num_samples 5000 \
#     --pairs_per_sentence 4 \
#     --domain_type "traffic_law" \
#     --language "en" \
#     --min_quality_score 0.60 \
#     --max_retries 4\
#     --shuffle_sentences \
#     --hf_token "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" \
#     --input_file "../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung.json" \
#     --output_file "../data/synthetic-data/stvo/qa_mistral_fast.jsonl"


# # ============================================================================
# # SCENARIO 3: Balanced Approach with NeuralHermes
# # ============================================================================
# # Best for: Quality-speed tradeoff
# # Expected: Quality scores 0.65-0.78, good speed
# # Time: ~1.5-3 hours for 5000 samples

# echo "=== SCENARIO 3: Balanced Generation with jpacifico/Chocolatine ==="

# python ../legal_qa_generation_multi_lingual.py \
#     --model_name "jpacifico/Chocolatine-14B-Instruct-DPO-v1.2" \
#     --num_samples 5000 \
#     --pairs_per_sentence 4 \
#     --domain_type "traffic_law" \
#     --language "en" \
#     --min_quality_score 0.65 \
#     --max_retries 4\
#     --shuffle_sentences \
#     --hf_token "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" \
#     --input_file "../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung.json" \
#     --output_file "../data/synthetic-data/stvo/qa_chocolatine_balanced.jsonl"


# # ============================================================================
# # SCENARIO 6: Testing Different Quality Thresholds
# # ============================================================================
# # Best for: Understanding quality-quantity tradeoff
# # Generates 4 versions with different quality requirements

# echo "=== SCENARIO 6: Quality Threshold Comparison ==="

# for threshold in 0.55 0.60 0.65 0.70; do
#     echo "Generating with quality threshold: $threshold"
#     python ../legal_qa_generation_multi_lingual.py \
#         --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
#         --num_samples 1000 \
#         --pairs_per_sentence 4 \
#         --domain_type "traffic_law" \
#         --language "en" \
#         --min_quality_score $threshold \
#         --max_retries 4\
#         --hf_token "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" \
#         --input_file "../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung.json" \
#         --output_file "../data/synthetic-data/stvo/qa_threshold_${threshold}_mistral.jsonl"
# done


# # ============================================================================
# # SCENARIO 7: Domain-Specific Multi-Run
# # ============================================================================
# # Generate data for different legal domains

# echo "=== SCENARIO 7: Multi-Domain Generation ==="

# for domain in "traffic_law" "regulatory" "contract"; do
#     echo "Generating for domain: $domain"
#     python ../legal_qa_generation_multi_lingual.py \
#         --model_name "meta-llama/Llama-2-7b-chat-hf" \
#         --num_samples 3000 \
#         --pairs_per_sentence 4 \
#         --domain_type "$domain" \
#         --language "en" \
#         --min_quality_score 0.65 \
#         --max_retries 4\
#         --hf_token "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" \
#         --input_file "../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung.json" \
#         --output_file "../data/synthetic-data/stvo/qa_domain_${domain}.jsonl"
# done



# ============================================================================
# SCENARIO 8: Incremental Generation (Resume-Friendly)
# ============================================================================
# Generate in smaller batches to manage resources and allow resumption

# echo "=== SCENARIO 8: Incremental Batch Generation ==="

# total_samples=10000
# batch_size=1000
# num_batches=$((total_samples / batch_size))

# for ((i=0; i<num_batches; i++)); do
#     echo "Batch $((i+1))/$num_batches - Generating $batch_size samples"
    
#     python ../legal_qa_generation_multi_lingual.py \
#         --model_name "meta-llama/Llama-2-7b-chat-hf" \
#         --num_samples $batch_size \
#         --pairs_per_sentence 4 \
#         --domain_type "traffic_law" \
#         --language "en" \
#         --min_quality_score 0.65 \
#         --max_retries 4\
#         --shuffle_sentences \
#         --hf_token "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" \
#         --input_file "../data/stvo/m2m100_418M_translated_main_content_Straßenverkehrs_Ordnung.json" \
#         --output_file "../data/synthetic-data/stvo/qa_batch_${i}.jsonl"
    
#     echo "Batch $((i+1)) complete. Sleeping 60 seconds..."
#     sleep 60
# done

# # Combine all batches
# echo "Combining all batches..."
# cat ../data/synthetic-data/stvo/qa_batch_*.jsonl > ../data/synthetic-data/stvo/qa_llama_incremental_total.jsonl 
# rm ../data/synthetic-data/stvo/qa_batch_*.jsonl



# ============================================================================
# SCENARIO 10: Quality Validation Script
# ============================================================================
# After generation, validate the dataset
# Validate the generated dataset
# python validate_qa_dataset.py "../data/synthetic-data/stvo/qa_llama_highquality.jsonl"





# ============================================================================
# NOTES FOR EXECUTION
# ============================================================================
# 
# 1. Update --hf_token with your actual Hugging Face token
# 2. Update file paths according to your directory structure
# 3. Uncomment the scenario you want to run
# 4. Monitor GPU memory: watch -n 1 nvidia-smi
# 5. Check logs in: ../data/synthetic-data/stvo/logs/
# 6. For long runs, use: nohup bash script.sh > execution.log 2>&1 &
#
# RECOMMENDED WORKFLOW:
# - Start with Scenario 2 (Mistral Fast) for testing
# - Then run Scenario 1 (Llama High Quality) for production
# - Validate outputs with Scenario 10
# - Use Scenario 4 to compare different translations
#
# EXPECTED QUALITY METRICS:
# - Mistral 7B: 0.60-0.72 average quality score
# - NeuralHermes: 0.65-0.78 average quality score  
# - Llama 2 7B: 0.70-0.82 average quality score
# - Llama 3 70B: 0.75-0.90 average quality score (if available)
#
# ============================================================================