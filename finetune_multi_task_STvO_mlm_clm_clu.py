from typing import Dict, List
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Assuming you have the datasets library installed
import random
import os
import argparse
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import logging  
from tqdm import tqdm
import transformers
from multi_task_classes_STvO_mlm_clm_clu_cls import  HierarchicalNSPDataset , LegalCLMDataset , UnifiedMultiTaskModel , TokenizerManager , HierarchicalClusteringDataset, CorrectIncorrectSFTDataset
from plotter import Plotter
import wandb
from tqdm.auto import tqdm # For a nice progress bar
from segmenter import LegalTextSegmenter
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)





#wandb login
# wandb_api_key = os.getenv("f34b3d1b1760e82ffef7ff865cc2c0d8ea49da15")
# wandb.login(key=wandb_api_key)


# =====================================================================================
# SECTION 1: UTILITY AND SETUP FUNCTIONS
# =====================================================================================

def set_seed(seed: int):
    """
    Sets the random seed for Python, PyTorch, and CUDA for reproducibility.
    Includes settings for deterministic GPU operations.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These are crucial for reproducible results on GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}. Deterministic GPU operations enabled.")


# =====================================================================================
# HUGGING FACE CHECKPOINTING
# ====================================================================================
def save_as_huggingface_checkpoint(model, tokenizer, output_dir: str, checkpoint_name: str):
    """
    Saves a fine-tuned UnifiedMultiTaskModel as a Hugging Face checkpoint
    in a dedicated subdirectory using SafeTensors format.

    Args:
        model: The trained UnifiedMultiTaskModel object.
        tokenizer: The tokenizer used for training.
        output_dir: The base directory to save all huggingface checkpoints.
        checkpoint_name: The name of the specific checkpoint to save.
    """
    try:
        from safetensors.torch import save_file
        use_safetensors = True
    except ImportError:
        logging.warning("safetensors not installed. Falling back to pytorch format.")
        use_safetensors = False
    
    # 1. Create a dedicated subdirectory for this specific checkpoint
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    logging.info(f"Saving model checkpoint to {checkpoint_path}")

    # 2. Save the model's configuration - FIX: Use base_encoder instead of lm_model
    config = model.base_encoder.config
    config.save_pretrained(checkpoint_path)

    # 3. Save the tokenizer
    tokenizer.save_pretrained(checkpoint_path)

    # 4. Save the model's weights - FIX: Save base_encoder state dict
    state_dict_to_save = model.base_encoder.state_dict()
    
    if use_safetensors:
        # Use SafeTensors format (recommended)
        # Make all tensors contiguous to avoid SafeTensors errors
        state_dict_contiguous = {k: v.contiguous() for k, v in state_dict_to_save.items()}
        output_model_path = os.path.join(checkpoint_path, "model.safetensors")
        save_file(state_dict_contiguous, output_model_path)
        logging.info(f"Saved model in SafeTensors format")
    else:
        # Fallback to PyTorch format
        output_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        torch.save(state_dict_to_save, output_model_path)
        logging.info(f"Saved model in PyTorch format")
    
    logging.info(f"Hugging Face checkpoint '{checkpoint_name}' saved successfully.")
    logging.info(f"Saved config type: {type(config).__name__}")


# =====================================================================================
# SECTION 2: CORE TRAINING FUNCTION
# =====================================================================================

def train(args, model, optimizer, scheduler, dataloaders,
           CLM_NSP_LOSS_LAMBDA, CLU_LOSS_LAMBDA, CLS_LOSS_LAMBDA, num_steps,
           device):
    """The main training loop, corrected and optimized."""
    training_history = {k: [] for k in ["epochs", "train_loss_epoch","steps", "combined_loss_step", "clm_loss_step","nsp_loss_step", "clu_loss_step", "cls_loss_step"]}
    global_step = 0

    print("Starting training...")
    print(f" Testing with : CLM_NSP_LOSS_LAMBDA: {CLM_NSP_LOSS_LAMBDA}, CLU_LOSS_LAMBDA: {CLU_LOSS_LAMBDA}, CLS_LOSS_LAMBDA: {CLS_LOSS_LAMBDA}")
    print(f"Number of total steps: {num_steps}")
    
    for epoch in range(args.num_epochs):
        model.train()
        iterators = {name: iter(loader) for name, loader in dataloaders.items() if 'val' not in name}
        # get batches safely from each iterator
        for name, iterator in iterators.items():
            if iterator is None:
                raise ValueError(f"DataLoader for task '{name}' is empty or not properly initialized.")
        total_epoch_loss = 0.0
        steps_with_loss = 0
        
        progress_bar = tqdm(range(num_steps), desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        
        for i in progress_bar:
            # --- Forward Passes for each available batch ---
            losses = {}
            if args.modeling_type == 'causal':
                core_task = 'clm'
                if (clm_batch := next(iterators[core_task], None)):
                    results = model(input_ids=clm_batch['input_ids'].to(device),
                                                      clm_labels=clm_batch['labels'].to(device), 
                                                      task_type=core_task)
                    losses[core_task] = results[f'{core_task}_loss']
                    
                    
            elif args.modeling_type == 'masked':
                core_task = 'nsp'
                if (nsp_batch := next(iterators[core_task], None)):
                    results= model(
                                    input_ids_a=nsp_batch['input_ids_a'].to(device),
                                    attention_mask_a=nsp_batch['attention_mask_a'].to(device),
                                    input_ids_b=nsp_batch['input_ids_b'].to(device),
                                    attention_mask_b=nsp_batch['attention_mask_b'].to(device),
                                    nsp_labels=nsp_batch['next_sentence_labels'].to(device), 
                                    task_type=core_task)
                    losses[core_task] = results[f'{core_task}_loss']

            if (clu_batch := next(iterators['clu'], None)):
                results= model(
                    input_ids=clu_batch['input_ids'].to(device),
                    attention_mask=clu_batch['attention_mask'].to(device),
                    positive_embedding=clu_batch['positive_embedding'].to(device),
                    negative_embedding=clu_batch['negative_embedding'].to(device),
                    task_type='clu'
                ) 
                losses['clu'] = results['clu_loss']

            if (cls_batch := next(iterators['cls'], None)):
                
                results = model(
                    input_ids=cls_batch['input_ids'].to(device),
                    attention_mask=cls_batch['attention_mask'].to(device),
                    cls_labels=cls_batch['labels'].to(device),
                    task_type='cls'
                )
                losses['cls'] = results['cls_loss']

            if not losses: continue
             
            # --- CORRECTED: Combine loss tensors directly ---
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            combined_loss = (CLM_NSP_LOSS_LAMBDA * losses.get(core_task, zero_loss) +
                             CLU_LOSS_LAMBDA * losses.get('clu', zero_loss) +
                             CLS_LOSS_LAMBDA * losses.get('cls', zero_loss)
            )

            if not isinstance(combined_loss, torch.Tensor) or torch.isnan(combined_loss) or torch.isinf(combined_loss):
                continue

            scaled_loss = combined_loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                # Only step scheduler after actual optimizer step
                if global_step % 1 == 0:  # Or your desired frequency
                    scheduler.step()
                progress_bar.set_postfix({'combined_step_loss': combined_loss.item(),
                                            f'{core_task}_loss': CLM_NSP_LOSS_LAMBDA * losses.get(core_task, zero_loss).item(),
                                            'clu_loss': CLU_LOSS_LAMBDA * losses.get('clu', zero_loss).item(),
                                            'cls_loss': CLS_LOSS_LAMBDA * losses.get('cls', zero_loss).item(),
                                            'global_step': global_step,
                                           'lr': scheduler.get_last_lr()[0]})
                # wandb.log({
                #     'train/step_weighted_loss': combined_loss.item(),
                #     f'train/step_{core_task}_loss (cross_entropy)': CLM_NSP_LOSS_LAMBDA * losses.get(core_task, zero_loss).item(),
                #     'train/step_clu_loss (cosine_embedding)': CLU_LOSS_LAMBDA * losses.get('clu', zero_loss).item(),
                #     'train/step_cls_loss (cross_entropy)': CLS_LOSS_LAMBDA * losses.get('cls', zero_loss).item(),
                # }, step=global_step)

            total_epoch_loss += combined_loss.item()
            steps_with_loss += 1
            # record step losses
            training_history["steps"].append(global_step)
            training_history["combined_loss_step"].append(combined_loss.item())
            training_history[f"{core_task}_loss_step"].append(losses.get(core_task, torch.tensor(0.0, device=device)).item())
            training_history["clu_loss_step"].append(losses.get('clu', torch.tensor(0.0, device=device)).item())
            training_history["cls_loss_step"].append(losses.get('cls', torch.tensor(0.0, device=device)).item())
            
        # --- End of Epoch: Validation and Checkpointing ---
        avg_train_loss = total_epoch_loss / steps_with_loss if steps_with_loss > 0 else 0
        # wandb.log({
        #     'epoch': epoch + 1,
        #     'train/avg_epoch_loss': avg_train_loss,
        # }, step=global_step)

        logger.info(f"Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Global Step: {global_step}")
        training_history["epochs"].append(epoch + 1)
        training_history["train_loss_epoch"].append(avg_train_loss)
        
        
    return training_history


# =====================================================================================
# SECTION 3: INFERENCE BLOCK
# =====================================================================================
def predict_clusters_hierarchical(
    new_sentences: List[Dict],
    model,
    tokenizer,
    paragraph_embeddings: dict,
    device: torch.device,
    max_segments: int = 64,  # New parameter for hierarchical model
    max_segment_length: int = 128, # New parameter
    top_k: int = 3
):
    """
    Predicts the top_k clusters for a list of new documents using a hierarchical model.
    This function tokenizes documents into segments before feeding them to the model.

    Args:
        new_sentences (list): A list of dictionaries, each with a 'text' key.
        model (MultiTaskModelClustering): The trained multi-task model.
        tokenizer (AutoTokenizer): The tokenizer used during training.
        paragraph_embeddings (dict): A dictionary mapping paragraph IDs to their
                                     pre-trained embedding vectors.
        device (torch.device): The device to run inference on ('cuda' or 'cpu').
        max_segments (int): The maximum number of segments (e.g., paragraphs) per document.
        max_segment_length (int): The maximum sequence length for each segment.
        top_k (int): The number of top clusters to return for each sentence.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              original sentence and its top_k predicted clusters (paragraphs) with scores.
    """
    model.eval()

    # Use SAME segmenter as training!
    segmenter = LegalTextSegmenter(tokenizer, max_segment_length)
    
    paragraph_ids = list(paragraph_embeddings.keys())
    paragraph_tensors = torch.tensor(
        list(paragraph_embeddings.values()), 
        dtype=torch.float
    ).to(device)
    
    with torch.no_grad():
        all_input_ids = []
        all_attention_masks = []
        
        for doc in new_sentences:
            text = doc['text']
            
            # FIX: Use proper segmentation
            paragraphs = segmenter.segment_text(text)
            
            # FIX: Warn if truncating
            if len(paragraphs) > max_segments:
                logger.warning(
                    f"Document has {len(paragraphs)} segments, "
                    f"truncating to {max_segments}"
                )
                paragraphs = paragraphs[:max_segments]

            
            # Tokenize each paragraph individually
            tokenized_paragraphs = tokenizer(
                paragraphs,
                padding='max_length',
                truncation=True,
                max_length=max_segment_length,
                return_tensors='pt'
            )
            
            # Get the tensors and ensure they are of the correct size
            # If a document has fewer than max_segments, we pad with empty tensors
            input_ids = tokenized_paragraphs['input_ids']
            attention_mask = tokenized_paragraphs['attention_mask']

            padded_input_ids = torch.zeros((max_segments, max_segment_length), dtype=torch.long)
            padded_attention_mask = torch.zeros((max_segments, max_segment_length), dtype=torch.long)

            num_actual_segments = input_ids.size(0)
            padded_input_ids[:num_actual_segments] = input_ids
            padded_attention_mask[:num_actual_segments] = attention_mask
            
            all_input_ids.append(padded_input_ids)
            all_attention_masks.append(padded_attention_mask)

        # Stack all documents into a single batch of shape (batch_size, max_segments, max_segment_length)
        inputs_ids_batch = torch.stack(all_input_ids).to(device)
        attention_mask_batch = torch.stack(all_attention_masks).to(device)
        
        # --- Get sentence embeddings from the hierarchical model ---
        # The model's forward pass now knows how to handle the 3D tensor
        results = model(
            input_ids=inputs_ids_batch,
            attention_mask=attention_mask_batch,
            task_type='clu'
        )
        sentence_embeddings = results['clu_logits']

        if sentence_embeddings is None:
            raise RuntimeError("Model returned None for clustering logits. Check task_type or forward method logic.")

        # --- Calculate cosine similarity and get top_k predictions ---
        # This part of the logic remains unchanged, as it operates on 2D tensors.
        sentence_embeddings_norm = F.normalize(sentence_embeddings, p=2, dim=1)
        paragraph_tensors_norm = F.normalize(paragraph_tensors, p=2, dim=1)
        cosine_similarities = torch.matmul(sentence_embeddings_norm, paragraph_tensors_norm.T)
        
        top_k_scores, top_k_indices = torch.topk(cosine_similarities, k=top_k, dim=1)

    # --- Format the results ---
    predictions = []
    for i, sentence in enumerate(new_sentences):
        sentence_preds = {
            "sentence": sentence['text'],
            "true_cluster": sentence.get('label', None),
            "predictions": []
        }
        for k in range(top_k):
            pred_idx = top_k_indices[i][k].item()
            pred_score = top_k_scores[i][k].item()
            pred_label = paragraph_ids[pred_idx]
            sentence_preds["predictions"].append({
                "cluster_id": pred_label,
                "similarity_score": pred_score,
            })
        predictions.append(sentence_preds)

    return predictions

# =====================================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Train a multi-task model for CLM and clustering.")
    parser.add_argument("--translator_model_name", type=str, default="m2m100_418M", help="Name of the translation model to use.")
    parser.add_argument("--clm_data_path", type=str, required=True, help="Path to the CLM dataset.")
    parser.add_argument("--nsp_data_path", type=str, required=True, help="Path to the MLM dataset.")
    parser.add_argument("--clu_data_path", type=str, required=True, help="Path to the clustering dataset.")
    parser.add_argument("--cls_data_path", type=str, required=True, help="Path to the classification dataset.")

    parser.add_argument("--paragraph_embeddings_file", type=str, default="entity_embeddings.json", help="Path to the entity embeddings file.")
    parser.add_argument("--modeling_type", type=str, default="causal", choices=['causal', 'masked'], help="Type of language modeling: 'causal' for CLM, 'masked' for MLM.") 
    parser.add_argument("--legal_text", type=str, default="STVO", choices=['STVO', 'BGB'], help="Type of legal text: 'STVO' or 'BGB'.")

    parser.add_argument("--train_history_path", type=str, default="training_history.json", help="Path to save the training history.")
    parser.add_argument("--plot_path", type=str, default="training_plot.png", help="Path to save the training plot.")
    parser.add_argument("--base_model_name", type=str, default="dbmdz/german-gpt2", help="Base model name from Hugging Face.")
    parser.add_argument("--hf_cache_dir", type=str, default="/fast_storage/siddig/driving-license/HF_models", help="Cache directory for Hugging Face models.")
    parser.add_argument("--output_dir", type=str, default="/fast_storage/siddig/driving-license/hybrid_models", help="Output directory for the trained model.")
    parser.add_argument("--hugg_out_dir", type=str, default="/fast_storage/siddig/driving-license/HF_models_stvo-final", help="Output directory for Hugging Face checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size_clm", type=int, default=16, help="Batch size for CLM.")
    parser.add_argument("--batch_size_clu", type=int, default=16, help="Batch size for classification.")
    parser.add_argument("--batch_size_nsp", type=int, default=16, help="Batch size for MLM.")
    parser.add_argument("--batch_size_cls", type=int, default=16, help="Batch size for classification.")

    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--clm_mlm_lambda", type=float, default=0.2, help="Weight for CLM loss in the combined loss.")
    parser.add_argument("--cls_lambda", type=float, default=0.1, help="Weight for classification loss in the combined loss.")
    parser.add_argument("--clm_nsp_lambdas", nargs='+', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help="List of weights for CLM loss in the combined loss.")
    parser.add_argument("--cls_lambdas", nargs='+', type=float, default=[0.1], help="List of weights for classification loss in the combined loss.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--clm_block_size", type=int, default=512, help="Max sequence length for CLM or block size.")
    parser.add_argument("--max_seq_length_clu", type=int, default=128, help="Max sequence length for clustering.")
    parser.add_argument("--max_seq_length_nsp", type=int, default=128, help="Max sequence length for MLM.")
    parser.add_argument("--max_seq_length_cls", type=int, default=128, help="Max sequence length for classification.")
    parser.add_argument("--max_segments", type=int, default=10, help="Max number of segments for hierarchical encoding.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--wandb_project", type=str, default="LLM-Drive-License-structured-Understanding-no_weights", help="WandB project name.")
    
    args = parser.parse_args()
    ### -- Print the arguments for debugging purposes ---
    print("Arguments Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("##############")

    # if no gpu found raise error
    if not torch.cuda.is_available():
        raise EnvironmentError("No GPU found. Please ensure that a compatible GPU is available and properly configured.")

    for clm_nsp_lambda in args.clm_nsp_lambdas:
        for cls_lambda in args.cls_lambdas:
            args.clm_nsp_lambda = clm_nsp_lambda
            args.cls_lambda = cls_lambda
        
            CLM_NSP_LOSS_LAMBDA = clm_nsp_lambda
            CLS_LOSS_LAMBDA = cls_lambda
            CLU_LOSS_LAMBDA = 1.0 - (CLM_NSP_LOSS_LAMBDA + CLS_LOSS_LAMBDA)
            
            # Checking combinations
            if CLM_NSP_LOSS_LAMBDA < 0 or CLM_NSP_LOSS_LAMBDA > 1:
                raise ValueError(f"Invalid CLM loss weight: {CLM_NSP_LOSS_LAMBDA}. Must be between 0 and 1.")

            if CLS_LOSS_LAMBDA < 0 or CLS_LOSS_LAMBDA > 1:
                raise ValueError(f"Invalid CLS loss weight: {CLS_LOSS_LAMBDA}. Must be between 0 and 1.")

            if CLU_LOSS_LAMBDA < 0:
                print(f"Skipping combination where CLU is not positive: CLM_NSP {CLM_NSP_LOSS_LAMBDA}, CLS {CLS_LOSS_LAMBDA}, CLU {CLU_LOSS_LAMBDA}")
                continue

            # Round to one decimal place for cleaner logging and file naming
            CLM_NSP_LOSS_LAMBDA = round(CLM_NSP_LOSS_LAMBDA, 1)
            CLS_LOSS_LAMBDA = round(CLS_LOSS_LAMBDA, 1)
            CLU_LOSS_LAMBDA = round(CLU_LOSS_LAMBDA, 1)

            # make sure the sum of all lambdas is 1.0
            if not abs((CLM_NSP_LOSS_LAMBDA + CLU_LOSS_LAMBDA + CLS_LOSS_LAMBDA) - 1.0) < 1e-6:
                raise ValueError(f"Sum of CLM, CLS, and CLU loss weights must be 1.0. "
                                    f"Got CLM: {CLM_NSP_LOSS_LAMBDA}, CLS: {CLS_LOSS_LAMBDA}, CLU: {CLU_LOSS_LAMBDA}. "
                                    f"Please adjust the weights accordingly.")
            
            model_name = args.base_model_name.split("/")[-1] # Extract model name from path

            # --- W&B Initialization (NEW) ---
            # Log hyperparameters and configurations to a W&B run
            # wandb_config = {
            #     **vars(args), # All command-line arguments are stored as a dict
            # }
            # wandb.init(
            #     project=args.wandb_project,
            #     config=wandb_config,
            #     name=f"{model_name}_{args.translator_model_name}_clm_nsp_{CLM_NSP_LOSS_LAMBDA}_clu_{CLU_LOSS_LAMBDA}_cls_{CLS_LOSS_LAMBDA}_epoch{args.num_epochs}",
            #     dir="/home/siddig/driving-license/wandb_logs"
            # )


            

            print(f"CLM_NSP Lambda: {CLM_NSP_LOSS_LAMBDA}, CLS Lambda: {CLS_LOSS_LAMBDA}, CLU Lambda: {CLU_LOSS_LAMBDA}")
            # --- Setup ---
            set_seed(args.seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Arguments: {args}")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # --- Tokenizer ---
            tokenizer_manager = TokenizerManager(access_token=os.environ.get("HF_TOKEN"))
            tokenizer = tokenizer_manager.initialize_tokenizer(args.base_model_name, args.hf_cache_dir, args.modeling_type)

            # --- Load Pre-trained Embeddings (Prerequisite) ---
            logger.info(f"Loading pre-trained paragraph embeddings from {args.paragraph_embeddings_file}...")
            # This should be a dictionary mapping paragraph IDs to tensors
            if not os.path.exists(args.paragraph_embeddings_file):
                raise FileNotFoundError(f"Paragraph embeddings file not found: {args.paragraph_embeddings_file}")
            with open(args.paragraph_embeddings_file, 'r', encoding='utf-8') as f:
                paragraph_embeddings = json.load(f)
            
            embedding_dim = len(next(iter(paragraph_embeddings.values()))) if paragraph_embeddings else 0
            print(f"Embedding dimension: {embedding_dim}")

            # --- Datasets and DataLoaders ---
            logger.info("Loading datasets...")
            # NOTE: It's best practice to run data splitting in a separate, one-time script.
            
            
            clu_dataset = HierarchicalClusteringDataset(
                filepath=args.clu_data_path,
                tokenizer=tokenizer,
                max_segments=args.max_segments,  # New argument
                max_segment_length=args.max_seq_length_clu,  # New argument
                paragraph_embeddings=paragraph_embeddings
            )

        
            cls_dataset = CorrectIncorrectSFTDataset(
                    filepath=args.cls_data_path,
                    tokenizer=tokenizer,
                    max_length=args.max_seq_length_cls
                )

            
            # --- Model Init ---
            logger.info("Initializing model...")
            
            model = UnifiedMultiTaskModel(
                base_model_name=args.base_model_name,
                access_token=tokenizer_manager.access_token,
                embedding_dim=embedding_dim,
                tokenizer_len=len(tokenizer),
                mask_token_id=tokenizer.mask_token_id, # Use tokenizer's mask token ID
                num_cls_labels=2,
                max_segments=args.max_segments,
                paragraph_embeddings=paragraph_embeddings,
                hf_cache_dir=args.hf_cache_dir,
                modeling_type=args.modeling_type,
            ).to(device)

            # --- Optimizer & Scheduler ---
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
            if args.modeling_type == 'causal':
                clm_dataset = LegalCLMDataset(args.clm_data_path, tokenizer, args.clm_block_size)
                dataloaders = {
                    'clm': DataLoader(clm_dataset, batch_size=args.batch_size_clm, shuffle=True, num_workers=0),
                    'clu': DataLoader(clu_dataset, batch_size=args.batch_size_clu, shuffle=True, num_workers=0),
                    'cls': DataLoader(cls_dataset, batch_size=args.batch_size_cls, shuffle=True, num_workers=0)
                }

                # data loaders length
                print(f"CLM Dataset Length: {len(clm_dataset)}, Clu Dataset Length: {len(clu_dataset)}, CLS Dataset Length: {len(cls_dataset)}")
                print(f"CLM DataLoader Length: {len(dataloaders['clm'])}, Clu DataLoader Length: {len(dataloaders['clu'])}, CLS DataLoader Length: {len(dataloaders['cls'])}")

            elif args.modeling_type == 'masked':
                nsp_dataset = HierarchicalNSPDataset(
                    data_source=args.nsp_data_path,
                    tokenizer=tokenizer,
                    max_segments=args.max_segments,
                    max_segment_length=args.max_seq_length_nsp
                )
                dataloaders = {
                    'nsp': DataLoader(nsp_dataset, batch_size=args.batch_size_nsp, shuffle=True, num_workers=0),
                    'clu': DataLoader(clu_dataset, batch_size=args.batch_size_clu, shuffle=True, num_workers=0),
                    'cls': DataLoader(cls_dataset, batch_size=args.batch_size_cls, shuffle=True, num_workers=0)
                }

                # data loaders length
                print(f"NSP Dataset Length: {len(nsp_dataset)}, Clu Dataset Length: {len(clu_dataset)}, CLS Dataset Length: {len(cls_dataset)}")
                print(f"NSP DataLoader Length: {len(dataloaders['nsp'])}, Clu DataLoader Length: {len(dataloaders['clu'])}, CLS DataLoader Length: {len(dataloaders['cls'])}")

                
            
            num_training_steps = args.num_epochs * max(len(loader) for loader in dataloaders.values())
            

                
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps)
            print(f"Total training steps: {num_training_steps}")
            
        

            # --- Run Training ---
            logger.info("--- Starting Training ---")
            training_history = train(
                args=args, model=model, optimizer=optimizer, scheduler=scheduler,
                dataloaders=dataloaders, device=device,
                CLM_NSP_LOSS_LAMBDA=CLM_NSP_LOSS_LAMBDA, CLU_LOSS_LAMBDA=CLU_LOSS_LAMBDA, CLS_LOSS_LAMBDA=CLS_LOSS_LAMBDA,
                  num_steps=num_training_steps
            )

            ################ Training Completed #####################
            # empty cache
            torch.cuda.empty_cache()

            # # report history in stats dir ../stats/stats-final
            # stats_dir = "../stats/stats-stvo-final"
            # if not os.path.exists(stats_dir):
            #     os.makedirs(stats_dir)
            # stats_file = os.path.join(stats_dir, f"{args.translator_model_name}_{args.base_model_name.split('/')[-1]}_{CLM_NSP_LOSS_LAMBDA}_{CLU_LOSS_LAMBDA}_{CLS_LOSS_LAMBDA}_{args.train_history_path}")
            # logging.info(f"Saving training history to {stats_file}")
            # with open(stats_file, 'w') as f:
            #     json.dump(training_history, f, indent=4)
            #     logging.info(f"Training history saved to {stats_file}")
            
            
            # ### --- Plot Training Progress ---
            
            # if training_history:
            #     # Plot training progress
            #     # Define the save path for the plot
            #     plot_dir = "../plots/plots-stvo-final"
            #     if not os.path.exists(plot_dir):
            #         os.makedirs(plot_dir)
            #     save_path = f"{plot_dir}/{args.translator_model_name}_{model_name}_{CLM_NSP_LOSS_LAMBDA}_{CLU_LOSS_LAMBDA}_{CLS_LOSS_LAMBDA}_{args.plot_path}"

            #     # Plot the training progress
            #     plotter = Plotter()
            #     plotter.plot_training_progress(args, training_history, save_path)

            # --- Save the Model ---
            output_dir = args.output_dir
            # You would save the final model state after training
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # use model name to save the model
            # logger.info
            # if args.legal_text == "STVO":
            torch.save(model.state_dict(), f"{output_dir}/{model_name}_{args.translator_model_name}_clm_nsp_{CLM_NSP_LOSS_LAMBDA}_clu_{CLU_LOSS_LAMBDA}_cls_{CLS_LOSS_LAMBDA}.pt")


            # HUGGING FACE CHECKPOINTING
            hugg_out_dir = args.hugg_out_dir
            # if args.legal_text == "BGB":
            save_as_huggingface_checkpoint(model, tokenizer, hugg_out_dir, f"{model_name}_{args.translator_model_name}_clm_nsp_{CLM_NSP_LOSS_LAMBDA}_clu_{CLU_LOSS_LAMBDA}_cls_{CLS_LOSS_LAMBDA}")

            logger.info("Training complete.")

            # # ---- RUN INFERENCE ----
            # logger.info("Running inference on new sentences...")
            # with open(args.clu_data_path, 'r', encoding='utf-8') as f:
            #     test_clu_data = json.load(f)
            # new_sentences = [{"text": item['text'], "label": item.get('label', None)} for item in test_clu_data]
            # predictions = predict_clusters_hierarchical(
            # new_sentences=new_sentences[:15],
            # model=model,
            # tokenizer=tokenizer,
            # paragraph_embeddings=paragraph_embeddings,
            # device=device,
            # max_segments=args.max_segments,  # These values should match your HierarchicalEncoder
            # max_segment_length=args.max_seq_length_clu,  # These values should match your HierarchicalEncoder
            # top_k=4
            #     )
            
            # # Print the predictions
            # for pred in predictions:
            #     print(f"Sentence: {pred['sentence']}")
            #     for cluster in pred['predictions']:
            #         print(f"  Cluster ID: {cluster['cluster_id']}, Similarity Score: {cluster['similarity_score']:.4f}")
            #     if 'true_cluster' in pred:
            #         print(f"  True Cluster: {pred['true_cluster']}")
            #     else:
            #         print("  True Cluster: Not provided")
            #     print("\n")


if __name__ == "__main__":
    main()
