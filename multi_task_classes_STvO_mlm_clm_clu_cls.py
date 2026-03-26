import logging
import os
import json
from typing import Any, Dict, List, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM , AutoModelForMaskedLM , BertForMaskedLM 
from transformers import AutoTokenizer
from torch import nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Any
from transformers.activations import ACT2FN # NEW IMPORT
from segmenter import LegalTextSegmenter
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F

import random


# ──────────────────────────────────────────────────────────────────────────────
# 1. Tokenizer Manager Class
# ──────────────────────────────────────────────────────────────────────────────

class TokenizerManager:
    """
    A class to manage tokenizer initialization and configuration for different modeling types.
    """
    
    def __init__(self, access_token=None):
        """
        Initialize the TokenizerManager.
        
        Args:
            access_token (str, optional): HuggingFace access token. If None, will use default token.
        """
        self.access_token = access_token 
        self.tokenizer = None

    def initialize_tokenizer(self, model_name=None, hf_cache_dir=None, modeling_type='causal', tokenizer=None):
        """
        Loads and dynamically configures the tokenizer for either a causal or masked model.
        
        Args:
            model_name (str): Name or path of the model to load tokenizer for
            cache_dir (str, optional): Directory to cache the tokenizer files
            modeling_type (str): Type of modeling - 'causal' or 'masked'
            
        Returns:
            AutoTokenizer: Configured tokenizer instance
        """
        # Load the tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=hf_cache_dir, 
                trust_remote_code=True, 
                token=self.access_token
        )
        
        # Configure based on modeling type
        if modeling_type == 'causal':
            self._configure_causal_tokenizer()
        elif modeling_type == 'masked':
            self._configure_masked_tokenizer()
        else:
            raise ValueError(f"Invalid modeling_type: {modeling_type}. Choose 'causal' or 'masked'.")
        
        # Final validation
        self._validate_tokenizer()
        
        return self.tokenizer
    
    def _configure_causal_tokenizer(self):
        """Configure tokenizer for causal language models."""
        self.tokenizer.padding_side = "left"
        
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logging.info("Setting pad_token to eos_token for a causal model.")
            else:
                logging.warning("No EOS token found. Causal model padding may be an issue.")
    
    def _configure_masked_tokenizer(self):
        """Configure tokenizer for masked language models."""
        self.tokenizer.padding_side = "right"  # Default is typically right
        
        # Add special tokens if they are not already present
        special_tokens_to_add = []
        
        if not hasattr(self.tokenizer, 'cls_token') or self.tokenizer.cls_token is None:
            special_tokens_to_add.append('<cls>')
        if not hasattr(self.tokenizer, 'sep_token') or self.tokenizer.sep_token is None:
            special_tokens_to_add.append('<sep>')
        if not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None:
            special_tokens_to_add.append('<mask>')
        
        if special_tokens_to_add:
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
            logging.info(f"Added {len(special_tokens_to_add)} special tokens: {special_tokens_to_add}")
    
    def _validate_tokenizer(self):
        """Validate the tokenizer configuration."""
        if self.tokenizer.pad_token is None:
            logging.warning("No pad token found and no fallback set. This may cause issues with batching.")
    
    def get_tokenizer_info(self):
        """
        Get information about the current tokenizer.
        
        Returns:
            dict: Dictionary containing tokenizer information
        """
        if self.tokenizer is None:
            return {"status": "No tokenizer initialized"}
        
        return {
            "vocab_size": self.tokenizer.vocab_size,
            "padding_side": getattr(self.tokenizer, 'padding_side', 'unknown'),
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token,
            "cls_token": getattr(self.tokenizer, 'cls_token', None),
            "sep_token": getattr(self.tokenizer, 'sep_token', None),
            "mask_token": getattr(self.tokenizer, 'mask_token', None)
        }
    
# ──────────────────────────────────────────────────────────────────────────────
# 2.  Data Loading and Preparation Classes
# ──────────────────────────────────────────────────────────────────────────────

# a. Causal Language Modeling Dataset Class
class LegalCLMDataset(Dataset):
    """PyTorch Dataset for Causal Language Modeling."""
    def __init__(self, filepath, tokenizer, block_size):
        """
        Loads text file, tokenizes sequentially, and formats into blocks for CLM.

        Args:
            filepath (str): Path to the text file.
            tokenizer (AutoTokenizer): The tokenizer to use.
            block_size (int): The size of each block for CLM training.
        """
        print(f"Loading and preparing CLM data from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                logging.info(f"Loaded CLM data from {filepath} with {len(text)} characters.")
        except FileNotFoundError:
            print(f"Error: CLM data file not found at {filepath}")
            # Initialize empty tensors if file not found
            self.input_ids = torch.tensor([], dtype=torch.long)
            self.labels = torch.tensor([], dtype=torch.long)
            return # Exit init early

        # 1. Tokenize the entire text sequentially without padding or truncation
        # Use return_tensors='pt' to get a tensor directly.
        tokenized_full_text = tokenizer(text, return_attention_mask=False, return_tensors='pt', padding=False, truncation=False)

        # We need the input_ids as a single 1D tensor
        if not tokenized_full_text or "input_ids" not in tokenized_full_text:
            print("Error: Unexpected tokenizer output format for CLM data.")
            self.input_ids = torch.tensor([], dtype=torch.long)
            self.labels = torch.tensor([], dtype=torch.long)
            return

        # Get the sequence tensor and flatten it to 1D
        all_token_ids = tokenized_full_text["input_ids"].squeeze(0) # Shape: (total_seq_len,)

        # Ensure we have a 1D tensor after squeezing
        if all_token_ids.ndim != 1:
             print(f"Error: Final token IDs tensor is not 1D after squeezing (shape: {all_token_ids.shape}).")
             self.input_ids = torch.tensor([], dtype=torch.long)
             self.labels = torch.tensor([], dtype=torch.long)
             return

        print(f"Tokenized full text into {len(all_token_ids)} tokens.")

        # 2. Split the long sequence into blocks
        # Discard the last part that doesn't form a full block
        num_tokens = len(all_token_ids)
        num_chunks = num_tokens // block_size
        if num_chunks == 0:
             print(f"Warning: Not enough tokens ({num_tokens}) to create even one block of size {block_size}. Returning empty dataset.")
             self.input_ids = torch.tensor([], dtype=torch.long)
             self.labels = torch.tensor([], dtype=torch.long)
             return # Return empty dataset if not enough tokens

        # Reshape into (num_chunks, block_size)
        self.input_ids = all_token_ids[:num_chunks * block_size].view(-1, block_size)

        # 3. Create labels by shifting inputs within each block
        self.labels = self.input_ids.clone()
        self.labels[:, :-1] = self.input_ids[:, 1:]
        self.labels[:, -1] = -100 # Standard practice to ignore the last token's prediction loss

        print(f"Prepared CLM dataset with {len(self)} blocks (sequences of size {block_size}).")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return input_ids and labels for a block
        return  {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]  # Use 'labels' key for consistency with Hugging Face datasets
        }


# 2.1  NSP + MLM Dataset Class
class NSPMLMDataset(Dataset):
    def __init__(self, data_source: Union[str, List[Dict[str, Any]]], tokenizer, max_length: int = 128):
        """
        Initializes the dataset, loading from a file path or a pre-processed list.
        
        Args:
            data_source (Union[str, List[Dict[str, Any]]]): Path to a .json or .jsonl file,
                                                           or a list of pre-processed NSP pairs.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            max_length (int): The maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self._load_data(data_source)

    def _load_data(self, data_source: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Handles loading data from a file or directly from a list."""
        if isinstance(data_source, list):
            logging.info("Using in-memory data for dataset.")
            return data_source
        
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"Error: Data file not found at {data_source}")
            
        # Try to load as a standard JSON file
        try:
            with open(data_source, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            logging.info(f"Loaded data from a standard JSON file: {data_source}")
            
        except json.JSONDecodeError:
            # Fallback to loading as a JSON Lines file
            raw_data = []
            with open(data_source, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        raw_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping malformed JSON line in {data_source}: {e}")
            logging.info(f"Loaded data from a JSON Lines file: {data_source}")

        # Ensure the labels are named correctly
        processed_data = []
        for item in raw_data:
            processed_data.append({
                'text_a': item['text_a'],
                'text_b': item['text_b'],
                'next_sentence_labels': item['label'] # Renamed for clarity
            })
        
        logging.info(f"Loaded {len(processed_data)} samples.")
        return processed_data

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single sample, tokenizing the pre-prepared sentence pair.
        """
        sample = self.processed_data[idx]
        
        # Tokenize the sentence pair
        encoding = self.tokenizer(
            sample['text_a'], 
            sample['text_b'],
            truncation=True,
            padding="max_length", # Use "max_length" to ensure consistent padding
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'next_sentence_labels': torch.tensor(sample['next_sentence_labels'], dtype=torch.long)
        }

# 2.2 Modified NSP + MLM Dataset Class
class HierarchicalNSPDataset(Dataset):
    def __init__(self, data_source: Union[str, List[Dict[str, Any]]], tokenizer, max_segments: int, max_segment_length: int):
        """
        Initializes the dataset for hierarchical NSP, loading from a file path.
        
        Args:
            data_source (Union[str, List[Dict[str, Any]]]): Path to a .json or .jsonl file.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            max_segments (int): The maximum number of segments (e.g., paragraphs) per document.
            max_segment_length (int): The maximum sequence length for each segment.
        """
        self.tokenizer = tokenizer
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        self.processed_data = self._load_data(data_source)
        self.segmenter = LegalTextSegmenter(tokenizer, max_segment_length=max_segment_length)
        logging.info(f"Loaded {len(self.processed_data)} samples for NSP.")

    def _load_data(self, data_source: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        # ... (The loading logic from your original script is fine and should be kept here) ...
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"Error: Data file not found at {data_source}")
        
        try:
            with open(data_source, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            logging.info(f"Loaded data from a standard JSON file: {data_source}")
        except json.JSONDecodeError:
            raw_data = []
            with open(data_source, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        raw_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping malformed JSON line in {data_source}: {e}")
            logging.info(f"Loaded data from a JSON Lines file: {data_source}")

        processed_data = []
        for item in raw_data:
            processed_data.append({
                'text_a': item['text_a'],
                'text_b': item['text_b'],
                'next_sentence_labels': item['label']
            })
        return processed_data

    def _tokenize_hierarchically(self, text: str):
        """Helper function to tokenize a single long document hierarchically."""
        # paragraphs = text.split('.')
        paragraphs = self.segmenter.segment_text(text)
        
        
        # Limit to max_segments
        paragraphs = paragraphs[:self.max_segments]
        
        
        tokenized = self.tokenizer(
            paragraphs,
            padding='max_length',
            truncation=True,
            max_length=self.max_segment_length,
            return_tensors='pt'
        )

        # Pad number of segments to a fixed size
        num_actual_segments = tokenized['input_ids'].size(0)
        padded_input_ids = torch.zeros((self.max_segments, self.max_segment_length), dtype=torch.long)
        padded_attention_mask = torch.zeros((self.max_segments, self.max_segment_length), dtype=torch.long)

        padded_input_ids[:num_actual_segments] = tokenized['input_ids']
        padded_attention_mask[:num_actual_segments] = tokenized['attention_mask']

        return padded_input_ids, padded_attention_mask


    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.processed_data[idx]
        
        # Tokenize both text_a and text_b hierarchically
        input_ids_a, attention_mask_a = self._tokenize_hierarchically(sample['text_a'])
        input_ids_b, attention_mask_b = self._tokenize_hierarchically(sample['text_b'])

        return {
            'input_ids_a': input_ids_a,
            'attention_mask_a': attention_mask_a,
            'input_ids_b': input_ids_b,
            'attention_mask_b': attention_mask_b,
            'next_sentence_labels': torch.tensor(sample['next_sentence_labels'], dtype=torch.long)
        }

# 3.1 Embedding Clustering Dataset Class
class EmbeddingClusteringDataset(Dataset):
    """
    PyTorch Dataset for metric learning on sentence-paragraph pairs.
    Provides an anchor (sentence), a positive embedding (correct paragraph),
    and a negative embedding (incorrect paragraph).
    """
    def __init__(self, filepath, tokenizer, max_seq_length, paragraph_embeddings: dict):
        """
        Args:
            filepath (str): Path to the clustering data file (e.g., cls_sentence_data.json).
            tokenizer: The Hugging Face tokenizer.
            max_seq_length (int): Maximum sequence length.
            paragraph_embeddings (dict): A pre-loaded dictionary mapping entity IDs to their pre-trained embeddings.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.paragraph_embeddings = paragraph_embeddings
        
        # Load the data (e.g., from JSON)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f) # Expects a list of {"text": "...", "label": "..."}
                logging.info(f"Loaded clustering data from {filepath}")
        except FileNotFoundError:
            print(f"Error: Clustering data file not found at {filepath}")
            self.data = []
            return
        except Exception as e:
            print(f"Error loading clustering data from {filepath}: {e}")
            self.data = []
            return

        # Get a list of all possible labels (paragraph IDs) to sample negatives from
        self.all_labels = list(set(item['label'] for item in self.data))

        # Store these for easy access, needed by train_multi_task_model
        self.num_classes = len(self.all_labels) # Number of unique paragraph IDs
        self.id_to_label = {i: label for i, label in enumerate(self.all_labels)} # Simplified map for consistency (not directly used by __getitem__)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor_text = item['text']
        positive_label = item['label']

        # Sample a negative label that is different from the positive one
        negative_label = positive_label
        # Added loop safety for very small all_labels set
        if len(self.all_labels) > 1: 
            while negative_label == positive_label:
                negative_label = random.choice(self.all_labels)
        else:
            # If only one label exists, can't find a different negative.
            # This case means triplet loss is not well-defined. Handle gracefully.
            # You might want to skip this example, or assign a dummy negative.
            # For now, let's just make negative_embedding a zero tensor if only one label
            raise ValueError(f"Only one unique label found: {positive_label}. Cannot create negative pairs for triplet loss.")
            # pass # Keep negative_label same as positive, loss will be 0 but not error out

        # Tokenize the anchor text
        anchor_encoding = self.tokenizer(
            anchor_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Get the pre-trained embeddings for the positive and negative labels
        # --- FIX: Ensure embeddings are 1D PyTorch tensors ---
        # They might be loaded as lists (e.g., [0.1, 0.2]) or nested lists ([[0.1, 0.2]])
        # Convert to tensor and squeeze to ensure it's 1D
        
        # Make sure positive_label and negative_label are valid keys in paragraph_embeddings
        if positive_label not in self.paragraph_embeddings:
            raise KeyError(f"Positive label '{positive_label}' not found in paragraph_embeddings. Check data consistency.")
        if negative_label not in self.paragraph_embeddings:
            raise KeyError(f"Negative label '{negative_label}' not found in paragraph_embeddings. Check data consistency.")


        positive_embedding_raw = self.paragraph_embeddings[positive_label]
        negative_embedding_raw = self.paragraph_embeddings[negative_label]

        # Convert to float tensor and squeeze to remove dimensions of size 1
        # Example: if raw is [[0.1, 0.2]], this becomes (1, D) then squeeze to (D,)
        positive_embedding = torch.tensor(positive_embedding_raw, dtype=torch.float).squeeze()
        negative_embedding = torch.tensor(negative_embedding_raw, dtype=torch.float).squeeze()

        # Final check for dimensionality (should be 1D after squeeze)
        if positive_embedding.ndim != 1 or negative_embedding.ndim != 1:
            raise ValueError(f"Embeddings for labels must be convertible to 1D tensors after squeeze(). "
                             f"Positive embedding shape: {positive_embedding.shape}, Negative embedding shape: {negative_embedding.shape}. "
                             f"Check paragraph_embeddings.json structure.")


        return {
            "input_ids": anchor_encoding["input_ids"].squeeze(0), # Remove batch dimension from tokenizer output
            "attention_mask": anchor_encoding["attention_mask"].squeeze(0), # Remove batch dimension
            "positive_embedding": positive_embedding,
            "negative_embedding": negative_embedding
        }
    
# 3.2 Modified EmbeddingClusteringDataset
class HierarchicalClusteringDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_segments, max_segment_length, paragraph_embeddings):
        # ... (init logic for loading self.data, self.all_labels, etc., remains the same) ...
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
                logging.info(f"Loaded clustering data from {filepath}")
        except FileNotFoundError:
            print(f"Error: Clustering data file not found at {filepath}")
            self.data = []
            return

        self.tokenizer = tokenizer
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        self.paragraph_embeddings = paragraph_embeddings
        self.all_labels = list(set(item['label'] for item in self.data))
        self.segmenter = LegalTextSegmenter(tokenizer, max_segment_length=max_segment_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor_text = item['text']
        positive_label = item['label']

        # --- Negative Sampling Logic (remains the same) ---
        negative_label = positive_label
        if len(self.all_labels) > 1:
            while negative_label == positive_label:
                negative_label = random.choice(self.all_labels)
        else:
            raise ValueError(f"Only one unique label found: {positive_label}. Cannot create negative pairs.")

        # --- NEW: On-the-fly Hierarchical Tokenization ---
        # Heuristic: split by double newline or period. Be consistent with inference!
        # paragraphs = anchor_text.split('.')
        paragraphs = self.segmenter.segment_text(anchor_text)
        
        
        # Limit to max_segments
        paragraphs = paragraphs[:self.max_segments]
        

        
        tokenized_paragraphs = self.tokenizer(
            paragraphs,
            padding='max_length',
            truncation=True,
            max_length=self.max_segment_length,
            return_tensors='pt'
        )
        
        input_ids = tokenized_paragraphs['input_ids']
        attention_mask = tokenized_paragraphs['attention_mask']

        # Correctly pad the number of segments with zero tensors
        padded_input_ids = torch.zeros((self.max_segments, self.max_segment_length), dtype=torch.long)
        padded_attention_mask = torch.zeros((self.max_segments, self.max_segment_length), dtype=torch.long)

        num_actual_segments = input_ids.size(0)
        padded_input_ids[:num_actual_segments] = input_ids
        padded_attention_mask[:num_actual_segments] = attention_mask
        
        # --- Embedding Retrieval (remains the same) ---
        positive_embedding = torch.tensor(self.paragraph_embeddings[positive_label], dtype=torch.float).squeeze()
        negative_embedding = torch.tensor(self.paragraph_embeddings[negative_label], dtype=torch.float).squeeze()
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "positive_embedding": positive_embedding,
            "negative_embedding": negative_embedding
        }
    


# 4 classification dataset :
class CorrectIncorrectSFTDataset(Dataset):
    def __init__(self, filepath: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(filepath)
        self.label_map = {"incorrect": 0, "correct": 1}
        logging.info(f"Loaded {len(self.data)} samples for correct/incorrect classification from {filepath}.")

    def _load_data(self, filepath: str) -> List[Dict[str, Any]]:
        # This is similar to your other data loading functions
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: Data file not found at {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Format the input string for classification
        # We combine the question and answer into a single input sequence
        input_text = f"Question: {sample['question']} [SEP] Answer: {sample['answer']}"
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.label_map[sample['label']], dtype=torch.long),
            'input_text': input_text # Include raw text for potential debugging or analysis
        }


############ # 4. Multi-Task Model Class ############ 
# multi-head attention layer that incorporates KG information
class KGAwareAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_kg_nodes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_kg_nodes = num_kg_nodes
        
        # Projection for KG embeddings if needed
        self.kg_projection = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanism to combine text and KG information
        self.kg_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, text_repr, kg_emb):
        """
        text_repr: (batch_size, hidden_size)
        kg_emb: (num_kg_nodes, hidden_size)
        """
        # Project KG embeddings to match hidden size if necessary
        kg_emb_proj = self.kg_projection(kg_emb) # (num_kg_nodes, hidden_size)
        
        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(text_repr, kg_emb_proj.t()) / (self.hidden_size ** 0.5) # (batch, num_nodes)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (batch, num_nodes)
        
        kg_context = torch.matmul(attn_weights, kg_emb_proj) # (batch, hidden_size)
        
        # Gating logic
        gate_input = torch.cat([text_repr, kg_context], dim=-1) # (batch, hidden*2)
        gate = self.kg_gate(gate_input) # (batch, hidden_size)
        
        enhanced_repr = gate * kg_context + (1 - gate) * text_repr
        return enhanced_repr, attn_weights
           
class UnifiedMultiTaskModel(nn.Module):
    """
    Unified architecture where all tasks share the same base encoder.
    This ensures lambda weights affect all downstream tasks.
    """
    
    def __init__(self, base_model_name, embedding_dim, num_cls_labels, hf_cache_dir, 
                 max_segments,paragraph_embeddings,use_kg_attention=True,num_kg_layers=2, **kwargs):
        super().__init__()
        
        self.paragraph_embeddings_dict = paragraph_embeddings  # Store the KG embeddings for use in the forward pass
        self.use_kg_attention = use_kg_attention
        # Single shared base encoder for ALL tasks
        self.base_encoder = AutoModel.from_pretrained(
            base_model_name, 
            cache_dir=hf_cache_dir,
            token=kwargs.get('access_token'),
        )

        self.hidden_size = self.base_encoder.config.hidden_size
        
        
        # KG COMPONENTS
        self.kg_projection = None
        if self.use_kg_attention and paragraph_embeddings:
            self._initialize_kg_embeddings(paragraph_embeddings, self.hidden_size)
        # 1.Simpler, more standard gating , single-head attention for KG integration
        self.kg_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

        # 2. Multiple layers of KG-aware attention if enabled (can be toggled with num_kg_layers)
        self.num_kg_layers = num_kg_layers
        self.kg_attention_layers = nn.ModuleList([
            KGAwareAttentionLayer(self.hidden_size, len(paragraph_embeddings)) for _ in range(num_kg_layers)
        ]) if self.use_kg_attention and paragraph_embeddings else None

    
        # Task-specific components on top of shared encoder
        # 1. Language modeling head (for CLM/MLM)
        self.lm_head = nn.Linear(self.hidden_size, self.base_encoder.config.vocab_size)
        
        # 2. Hierarchical layer for document-level tasks
        self.hierarchical_layer = self._build_hierarchical_layer(
            self.hidden_size, max_segments
        )
        
        # 3. Task-specific heads
        self.clu_head = nn.Linear(self.hidden_size, embedding_dim)
        self.cls_head = nn.Linear(self.hidden_size, num_cls_labels)
        self.nsp_head = nn.Linear(self.hidden_size * 2, 2)
        
        # Loss functions
        self.clu_criterion = nn.CosineEmbeddingLoss(margin=0.5)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.nsp_criterion = nn.CrossEntropyLoss()
    
    def _initialize_kg_embeddings(self, paragraph_embeddings_dict, hidden_size):
        """Convert paragraph embeddings to a registered buffer for GPU efficiency."""
        paragraph_ids = sorted(paragraph_embeddings_dict.keys())
        self.paragraphs_ids = paragraph_ids  # Store the order of paragraph IDs for consistent retrieval
        embedding_list = []
        
        for pid in paragraph_ids:
            emb = paragraph_embeddings_dict[pid]
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            embedding_list.append(emb)
        
        raw_embeddings = torch.stack(embedding_list)
        input_dim = raw_embeddings.shape[1]
        
        # Register the raw embeddings as a buffer (not a parameter)
        # This ensures they move to GPU with .to(device) but aren't trained
        self.register_buffer('kg_embeddings_raw', raw_embeddings)
        
        # Initialize projection if dimensions don't match
        if input_dim != hidden_size:
            self.kg_projection = nn.Linear(input_dim, hidden_size)
        
        logging.info(f"Initialized KG with {len(paragraph_ids)} nodes.")

    def _get_projected_kg(self):
        """Helper to get projected KG embeddings on the fly."""
        if self.kg_projection is not None:
            return self.kg_projection(self.kg_embeddings_raw)
        return self.kg_embeddings_raw

    # Single-head attention layer that incorporates KG information, can be stacked for deeper integration
    def _kg_aware_attention(self, text_repr):
        """
        Efficient KG attention.
        text_repr: (batch_size, hidden_size)
        """
        kg_emb = self._get_projected_kg() # (num_kg_nodes, hidden_size)
        hidden_size = text_repr.shape[-1]
        
        # Scaled Dot-Product Attention
        # (batch, hidden) @ (hidden, num_nodes) -> (batch, num_nodes)
        attn_scores = torch.matmul(text_repr, kg_emb.t()) / (hidden_size ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # (batch, num_nodes) @ (num_nodes, hidden) -> (batch, hidden)
        kg_context = torch.matmul(attn_weights, kg_emb)
        
        # Gating logic
        gate_input = torch.cat([text_repr, kg_context], dim=-1)
        gate = self.kg_gate(gate_input)
        
        enhanced_repr = gate * kg_context + (1 - gate) * text_repr
        return enhanced_repr, attn_weights

    def _build_hierarchical_layer(self, hidden_size, max_segments):
        """Build the second-level transformer for document encoding"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    def _encode_segments(self, input_ids, attention_mask):
        """
        Encode document segments hierarchically using shared base encoder.
        Input shape: (batch, num_segments, segment_length)
        Output shape: (batch, hidden_size)
        """
        batch_size, n_segments, seg_len = input_ids.shape
        
        # Flatten to process all segments
        input_ids_flat = input_ids.view(-1, seg_len)
        attention_mask_flat = attention_mask.view(-1, seg_len)
        
        # Encode with shared base encoder
        outputs = self.base_encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        
        # Get segment representations (CLS token or mean pooling)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            segment_reprs = outputs.pooler_output
        else:
            segment_reprs = outputs.last_hidden_state[:, 0, :]
        
        # Reshape back to segments
        segment_reprs = segment_reprs.view(batch_size, n_segments, -1)
        
        # Apply hierarchical transformer
        seg_mask = (input_ids.sum(dim=-1) == 0)  # True for padding
        doc_repr = self.hierarchical_layer(
            segment_reprs,
            src_key_padding_mask=seg_mask
        )
        
        # Pool to document level
        doc_repr = doc_repr.masked_fill(seg_mask.unsqueeze(-1), -1e9).max(dim=1)[0]
        
        return doc_repr

    

    def forward(self, task_type, input_ids=None, attention_mask=None,
                clm_labels=None, cls_labels=None, nsp_labels=None,
                positive_embedding=None, negative_embedding=None,
                input_ids_a=None, attention_mask_a=None,
                input_ids_b=None, attention_mask_b=None):
        """
        Unified forward pass where all tasks use the shared base encoder.
        """
        result = {
            'clm_loss': None, 'clm_logits': None,
            'clu_loss': None, 'clu_logits': None,
            'cls_loss': None, 'cls_logits': None,
            'nsp_loss': None, 'nsp_logits': None,
            'kg_attn_weights': None # for interpretability of KG attention in CLS task
        }
        
        # === CLM Task ===
        if task_type == 'clm':
            # Use shared encoder + LM head
            outputs = self.base_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = self.lm_head(outputs.last_hidden_state)
            result['clm_logits'] = logits
            
            if clm_labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = clm_labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                result['clm_loss'] = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
        
        # === CLU Task ===
        elif task_type == 'clu':
            # Hierarchical encoding using shared base encoder
            if input_ids.dim() == 3:  # Already segmented
                doc_repr = self._encode_segments(input_ids, attention_mask)
            else:  # Flat sequence
                outputs = self.base_encoder(input_ids, attention_mask)
                doc_repr = outputs.last_hidden_state.mean(dim=1)
            
            embedding = self.clu_head(doc_repr)
            result['clu_logits'] = embedding
            
            if positive_embedding is not None and negative_embedding is not None:
                pos_loss = self.clu_criterion(
                    embedding, positive_embedding,
                    torch.ones(embedding.size(0), device=embedding.device)
                )
                neg_loss = self.clu_criterion(
                    embedding, negative_embedding,
                    -torch.ones(embedding.size(0), device=embedding.device)
                )
                result['clu_loss'] = pos_loss + neg_loss
        
        # # === CLS Task ===
        # elif task_type == 'cls':
        #     # Use shared encoder (same representations as CLM!)
        #     if input_ids.dim() == 3:  # Hierarchical
        #         text_rep = self._encode_segments(input_ids, attention_mask)
        #     else:  # Flat
        #         outputs = self.base_encoder(input_ids, attention_mask)
        #         text_rep = outputs.last_hidden_state.mean(dim=1)
            
        #     logits = self.cls_head(text_rep)
        #     result['cls_logits'] = logits
            
        #     if cls_labels is not None:
        #         result['cls_loss'] = self.cls_criterion(logits, cls_labels)

        # === CLS Task with KG Attention ===
        elif task_type == 'cls':
            outputs = self.base_encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :] # CLS token
            
            if self.use_kg_attention:
                # enhanced_rep, weights = self._kg_aware_attention(pooled)
                enhanced_rep, weights = self.kg_attention_layers[0](pooled, self._get_projected_kg())
                result['kg_attn_weights'] = weights
                logits = self.cls_head(enhanced_rep)
            else:
                logits = self.cls_head(pooled)

            result['cls_logits'] = logits
            if cls_labels is not None:
                result['cls_loss'] = self.cls_criterion(logits, cls_labels)


        # === NSP Task ===
        elif task_type == 'nsp':
            # Encode both documents with shared encoder
            repr_a = self._encode_segments(input_ids_a, attention_mask_a)
            repr_b = self._encode_segments(input_ids_b, attention_mask_b)
            
            combined = torch.cat([repr_a, repr_b], dim=-1)
            logits = self.nsp_head(combined)
            result['nsp_logits'] = logits
            
            if nsp_labels is not None:
                result['nsp_loss'] = self.nsp_criterion(logits, nsp_labels)
        
        return result




