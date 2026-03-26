import argparse
import logging
import os
import pandas as pd
import torch
import random
import numpy as np
from driving_license_data_loader import DrivingLicenseDataHandler
from driving_examiner import DrivingLicenseExaminer

# Configure logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If you are using GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # Ensure reproducibility when using CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main function to evaluate a language model on a driving license question dataset.
    """
    parser = argparse.ArgumentParser(description="Evaluate a language model on a driving license question dataset.")

    # Data and Model
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test data file.")
    parser.add_argument("--paragraph_embeddings_file", type=str, default="entity_embeddings.json", help="Path to the entity embeddings file.")
    parser.add_argument("--data_name", type=str, required=True, help="Type of the test data (german, irish, australia).")
    parser.add_argument("--model_checkpoint_dir", type=str, default="../data/multitask_models", help="Directory where the model checkpoints are stored.")
    parser.add_argument("--model_checkpoint", type=str, default="final", help="Name of the model checkpoint to use.")
    parser.add_argument("--baseline_model", type=str, required=True, help="Path to the baseline model checkpoint.")

    # Evaluation Parameters
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument('--decoding', type=str, default=None, help="Decoding strategy to use (e.g., greedy, beam).")
    parser.add_argument('--batch_samples', type=int, default=20, help="Number of samples to process in each batch.")
    # parser.add_argument('--prompt_versions', nargs='+', type=int, default=[1, 4], help='List of prompt versions to evaluate.')
    parser.add_argument('--prompt_version', type=int, default=4, help='Prompt version to evaluate.')
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0.1, 0.2], help='List of temperatures to evaluate.')
    parser.add_argument('--skip_images', action='store_true', help='If set, skip processing images.')
    parser.add_argument('--skip_category', type=str, default="direct_answer", help="Remove a specific category from processing.")

    # Evaluation Mode
    parser.add_argument('--eval_base', action='store_true', help='If set, evaluate the base model.')

    # TASK TYPE
    parser.add_argument('--task_type', type=str, default='clm', help='Task type for the model (e.g., clm, mlm).')
    parser.add_argument('--modeling_type', type=str, default='causal', help='Modeling type (e.g., causal, masked).')
    parser.add_argument('--num_cls_labels', type=int, default=2, help='Number of classification labels.')

    # SEQUENCE LENGTH
    parser.add_argument('--max_seq_length_clm', type=int, default=512, help='Maximum sequence length for CLM task.')
    parser.add_argument('--max_seq_length_clu', type=int, default=128, help='Maximum sequence length for clu task.')
    parser.add_argument('--max_seq_length_cls', type=int, default=128, help='Maximum sequence length for CLS task.')
    parser.add_argument('--max_seq_length_nsp', type=int, default=128, help='Maximum sequence length for NSP task.')
    parser.add_argument('--max_num_segments', type=int, default=5, help='Number of segments for hierarchical models.')

    # Generation Parameters
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum number of new tokens to generate.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling probability.")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling probability.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty for sampling.")

    # similarity threshold 
    parser.add_argument("--similarity_threshold", type=float, default=0.6, help="Similarity threshold for RAG.")

    # Output
    parser.add_argument("--results_file", type=str, default="results.csv", help="File to save the evaluation results.")
    parser.add_argument("--cache_dir", type=str, default="/fast_storage/siddig/driving-license/HF_models", help="Directory to cache models.")

    # Rag Parameters
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for RAG.")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="Chunk overlap for RAG.")
    parser.add_argument("--max_length_embedding", type=int, default=500, help="Maximum length for embedding.")
    parser.add_argument("--batch_size_embedding", type=int, default=16, help="Batch size for embedding.")
    parser.add_argument("--top_k_retriever", type=int, default=3, help="Top-k retriever for RAG.")

    args = parser.parse_args()
    set_seed(args.seed)

    # Log evaluation parameters
    logging.info("-" * 50)
    logging.info("Evaluation Parameters:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info(f"Device: {torch.cuda.get_device_name(0)}")
    logging.info(f"Device Used: {torch.cuda.current_device()}")
    logging.info("-" * 50)

    # Read the test data
    data_handler = DrivingLicenseDataHandler()
    if args.data_name == "german":
        questions_answers = data_handler.read_german_questions_answers(args.data_path, skip_category=args.skip_category, skip_images=args.skip_images)
        logger.info(f"Loaded {len(questions_answers)} questions and answers for German.")
    elif args.data_name == "irish":
        questions_answers = data_handler.read_irish_questions_answers(args.data_path, skip_images=args.skip_images)
        logger.info(f"Loaded {len(questions_answers)} questions and answers for Irish.")
    elif args.data_name == "austrian":
        questions_answers = data_handler.read_austrian_questions_answers(args.data_path, language='en', skip_images=args.skip_images)
        logger.info(f"Loaded {len(questions_answers)} questions and answers for Austrian.")
    else:
        logging.error(f"Unsupported data name: {args.data_name}")
        return

    # Load baseline model
    logging.info(f"Loading baseline model from {args.baseline_model}")
    examiner = DrivingLicenseExaminer(args)

    results_df = pd.DataFrame()  # Initialize an empty DataFrame to store results

    
    logging.info(f"Evaluating with prompt version: {args.prompt_version}")
    stats = examiner.evaluate_batch_model(questions_answers, inference_batch_size=args.batch_samples)

    logging.info("Baseline Model Metrics:")
    logging.info(f"Accuracy: {stats['accuracy']}")
    logging.info(f"Precision: {stats['precision']}")
    logging.info(f"Recall: {stats['recall']}")
    logging.info(f"F1 Macro: {stats['f1_macro']}")
    logging.info(f"F1 Micro: {stats['f1_micro']}")
    logging.info(f"MCC: {stats['MCC']}")
    logging.info("-" * 50)
    model_base_name = args.baseline_model.split("/")[-1] if "/" in args.baseline_model else args.baseline_model
    if args.eval_base:
        model_name = f"baseline_{model_base_name}"
    else:
        model_name = f"{model_base_name}_{args.model_checkpoint}"
    # Record baseline metrics
    baseline_df = pd.DataFrame({
        "Model_Name": [model_name],
        "data_name": [args.data_name],
        "Number_of_Questions": [stats["number_of_questions"]],
        "Number_of_Subquestions": [stats["number_of_subquestions"]],
        "number_of_true_yes": [stats["number_true_yes"]],
        "number_of_true_no": [stats["number_true_no"]],
        "number_of_pred_yes": [stats["number_pred_yes"]],
        "number_of_pred_no": [stats["number_pred_no"]],
        "number_of_pred_invalid": [stats["number_pred_invalid"]],
        "Accuracy": [stats["accuracy"]],
        "Balanced_Accuracy": [stats["balanced_accuracy"]],
        "Precision": [stats["precision"]],
        "Recall": [stats["recall"]],
        "F1_Macro": [stats["f1_macro"]],
        "F1_Micro": [stats["f1_micro"]],
        "MCC": [stats["MCC"]],
        "similarity_threshold": [args.similarity_threshold],
        "Seed": [args.seed],
        "prompt_version": [args.prompt_version],
        "MAX_New_Tokens": [args.max_new_tokens],
        "Max_Length": [args.max_seq_length],
        "Top_P": [args.top_p],
        "Top_K": [args.top_k],
        "Temperature": [args.temperature],
        "Repetition_Penalty": [args.repetition_penalty],
        "Batch samples": [args.batch_samples],
        "evaluation_mode": ["non-positive-no"],
        "Decoding Strategy": [args.decoding],
        "Skip Category": [args.skip_category],
        "Skip Images": [args.skip_images],
        "Chunk Size": [args.chunk_size],
        "Chunk Overlap": [args.chunk_overlap],
        "Max Length Embedding": [args.max_length_embedding],
        "Batch Size Embedding": [args.batch_size_embedding],
        "Top K Retriever": [args.top_k_retriever]

    })
    results_df = pd.concat([results_df, baseline_df], ignore_index=True)

    # Save the combined results
    base_path, extension = os.path.splitext(args.results_file)
    new_path = f"{base_path}{extension}"

    try:
        existing_df = pd.read_csv(new_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    except FileNotFoundError:
        logging.info("No existing results file found, creating a new one.")

    results_df.to_csv(new_path, index=False)
    logging.info(f"Evaluation results saved successfully to {new_path}")

if __name__ == "__main__":
    main()
