import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm.auto import tqdm
import argparse
import os
import random
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import re
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """
    Enhanced PyTorch Dataset for loading triplets from a text file.
    Includes data validation, statistics, and better error handling.
    """
    def __init__(self, filepath: str, delimiter: str = '\t', validation_split: float = 0.1):
        self.triplets = []
        self.entities = set()
        self.relations = set()
        self.validation_split = validation_split
        
        # Statistics tracking
        self.stats = defaultdict(int)
        
        logger.info(f"Reading triplets from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
                
            try:
                parts = line.split(delimiter)
                if len(parts) != 3:
                    logger.warning(f"Line {line_num}: Expected 3 parts, got {len(parts)}")
                    self.stats['malformed_lines'] += 1
                    continue
                    
                h, r, t = [part.strip() for part in parts]
                
                # Validate that parts are not empty
                if not all([h, r, t]):
                    logger.warning(f"Line {line_num}: Empty components found")
                    self.stats['empty_components'] += 1
                    continue
                
                self.triplets.append((h, r, t))
                self.entities.add(h)
                self.entities.add(t)
                self.relations.add(r)
                self.stats['valid_triplets'] += 1
                
            except Exception as e:
                logger.warning(f"Line {line_num}: Error processing line - {e}")
                self.stats['processing_errors'] += 1
        
        if not self.triplets:
            raise ValueError("No valid triplets found in the dataset")
        
        # Create mappings from string to integer ID
        self.entity_list = sorted(list(self.entities))
        self.relation_list = sorted(list(self.relations))
        
        self.entity_to_id = {e: i for i, e in enumerate(self.entity_list)}
        self.relation_to_id = {r: i for i, r in enumerate(self.relation_list)}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}
        
        self.num_entities = len(self.entity_list)
        self.num_relations = len(self.relation_list)
        
        # Split data into train/validation
        self._create_train_val_split()
        
        # Log statistics
        self._log_statistics()

    def _create_train_val_split(self):
        """Create train/validation split while maintaining data integrity."""
        random.shuffle(self.triplets)
        split_idx = int(len(self.triplets) * (1 - self.validation_split))
        
        self.train_triplets = self.triplets[:split_idx]
        self.val_triplets = self.triplets[split_idx:]
        
        logger.info(f"Split: {len(self.train_triplets)} train, {len(self.val_triplets)} validation")

    def _log_statistics(self):
        """Log comprehensive dataset statistics."""
        logger.info(f"Dataset Statistics:")
        logger.info(f"  - Total entities: {self.num_entities}")
        logger.info(f"  - Total relations: {self.num_relations}")
        logger.info(f"  - Total triplets: {len(self.triplets)}")
        logger.info(f"  - Valid triplets: {self.stats['valid_triplets']}")
        logger.info(f"  - Malformed lines: {self.stats['malformed_lines']}")
        logger.info(f"  - Empty components: {self.stats['empty_components']}")
        logger.info(f"  - Processing errors: {self.stats['processing_errors']}")
        
        # Relation frequency analysis
        relation_counts = defaultdict(int)
        for _, r, _ in self.triplets:
            relation_counts[r] += 1
        
        logger.info(f"  - Most frequent relations:")
        for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"    {rel}: {count}")

    def get_train_data(self):
        """Return training triplets as integer IDs."""
        return [(self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]) 
                for h, r, t in self.train_triplets]

    def get_val_data(self):
        """Return validation triplets as integer IDs."""
        return [(self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]) 
                for h, r, t in self.val_triplets]

    def __len__(self):
        return len(self.train_triplets)

    def __getitem__(self, idx):
        h, r, t = self.train_triplets[idx]
        return (
            self.entity_to_id[h],
            self.relation_to_id[r],
            self.entity_to_id[t]
        )


class TransE(nn.Module):
    """
    Enhanced implementation of the TransE model with improved initialization,
    regularization, and scoring functions.
    """
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, 
                 margin: float = 1.0, l2_reg: float = 0.01, dropout: float = 0.1):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.l2_reg = l2_reg

        # Embedding layers for entities and relations
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Improved initialization
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embeddings with improved strategy."""
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # Normalize entity embeddings to unit sphere
        self.entity_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)
        
        # Scale relation embeddings appropriately
        self.relation_embeddings.weight.data *= 0.5

    def _normalize_embeddings(self):
        """Normalize entity embeddings to have unit L2 norm."""
        with torch.no_grad():
            # Clamp to avoid numerical issues
            self.entity_embeddings.weight.data.clamp_(-1, 1)
            self.entity_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate the TransE score for a batch of triplets.
        Score is the L2 distance: ||h + r - t||
        """
        # Normalize entity embeddings before each forward pass
        self._normalize_embeddings()
        
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        
        # Apply dropout during training
        if self.training:
            h_emb = self.dropout(h_emb)
            r_emb = self.dropout(r_emb)
            t_emb = self.dropout(t_emb)

        # Calculate the score using L2 norm
        score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
        return score

    def get_regularization_loss(self) -> torch.Tensor:
        """Calculate L2 regularization loss."""
        entity_reg = torch.norm(self.entity_embeddings.weight, p=2)**2
        relation_reg = torch.norm(self.relation_embeddings.weight, p=2)**2
        return self.l2_reg * (entity_reg + relation_reg) / (self.num_entities + self.num_relations)


class NegativeSampler:
    """
    Enhanced negative sampling with multiple strategies.
    """
    def __init__(self, num_entities: int, strategy: str = 'uniform'):
        self.num_entities = num_entities
        self.strategy = strategy

    def sample(self, batch_size: int, device: torch.device, 
               positive_heads: Optional[torch.Tensor] = None,
               positive_tails: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate negative samples using different strategies.
        Returns corrupted heads and tails.
        """
        if self.strategy == 'uniform':
            corrupted_h = torch.randint(0, self.num_entities, (batch_size,), device=device)
            corrupted_t = torch.randint(0, self.num_entities, (batch_size,), device=device)
        
        elif self.strategy == 'type_aware' and positive_heads is not None and positive_tails is not None:
            # Avoid corrupting with the same entity (basic type awareness)
            corrupted_h = torch.randint(0, self.num_entities, (batch_size,), device=device)
            corrupted_t = torch.randint(0, self.num_entities, (batch_size,), device=device)
            
            # Ensure we don't corrupt with the original entities
            mask_h = corrupted_h == positive_heads
            mask_t = corrupted_t == positive_tails
            
            if mask_h.any():
                corrupted_h[mask_h] = torch.randint(0, self.num_entities, (mask_h.sum(),), device=device)
            if mask_t.any():
                corrupted_t[mask_t] = torch.randint(0, self.num_entities, (mask_t.sum(),), device=device)
        
        else:
            corrupted_h = torch.randint(0, self.num_entities, (batch_size,), device=device)
            corrupted_t = torch.randint(0, self.num_entities, (batch_size,), device=device)
        
        return corrupted_h, corrupted_t


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_model(model: TransE, dataset: TripletDataset, device: torch.device, 
                  batch_size: int = 128) -> Dict[str, float]:
    """Evaluate the model on validation data."""
    model.eval()
    val_data = dataset.get_val_data()
    
    if not val_data:
        return {'val_loss': float('inf')}
    
    total_loss = 0.0
    num_batches = 0
    negative_sampler = NegativeSampler(dataset.num_entities)
    criterion = nn.MarginRankingLoss(margin=model.margin, reduction='mean')
    
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size]
            if not batch:
                continue
                
            h = torch.tensor([t[0] for t in batch], device=device)
            r = torch.tensor([t[1] for t in batch], device=device)
            t = torch.tensor([t[2] for t in batch], device=device)
            
            # Generate negative samples
            corrupted_h, corrupted_t = negative_sampler.sample(len(batch), device, h, t)
            
            # Calculate scores
            positive_scores = model(h, r, t)
            negative_scores_h = model(corrupted_h, r, t)
            negative_scores_t = model(h, r, corrupted_t)
            
            # Average negative scores
            negative_scores = (negative_scores_h + negative_scores_t) / 2
            
            # Calculate loss
            target = torch.tensor([-1], dtype=torch.float, device=device)
            loss = criterion(positive_scores, negative_scores, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return {'val_loss': total_loss / num_batches if num_batches > 0 else float('inf')}


def plot_training_history(train_losses: List[float], val_losses: List[float], 
                         output_dir: str):
    """Plot and save training history."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(train_losses) > 10:
        # Show smoothed version for better visualization
        window = max(1, len(train_losses) // 20)
        smoothed_train = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        plt.plot(smoothed_train, label=f'Smoothed Training (window={window})', color='blue')
        
        if val_losses and len(val_losses) > 10:
            smoothed_val = np.convolve(val_losses, np.ones(window)/window, mode='valid')
            plt.plot(smoothed_val, label=f'Smoothed Validation (window={window})', color='red')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Smoothed Training History')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_model_artifacts(model: TransE, dataset: TripletDataset, args, 
                        train_losses: List[float], val_losses: List[float]):
    """Save model, embeddings, and metadata."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_entities': dataset.num_entities,
        'num_relations': dataset.num_relations,
        'embedding_dim': args.embedding_dim,
        'margin': args.margin,
        'l2_reg': args.l2_reg,
        'dropout': args.dropout,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'entity_to_id': dataset.entity_to_id,
        'relation_to_id': dataset.relation_to_id,
        'id_to_entity': dataset.id_to_entity,
        'id_to_relation': dataset.id_to_relation,
    }
    
    torch.save(checkpoint, output_dir / 'model_checkpoint.pt')
    
    # Save embeddings separately for easy access
    entity_embeddings_map = {
        entity: model.entity_embeddings.weight.data[idx].cpu().numpy()
        for entity, idx in dataset.entity_to_id.items()
    }
    
    relation_embeddings_map = {
        relation: model.relation_embeddings.weight.data[idx].cpu().numpy()
        for relation, idx in dataset.relation_to_id.items()
    }
    
    torch.save(entity_embeddings_map, output_dir / args.save_filename)
    torch.save(relation_embeddings_map, output_dir / 'relation_embeddings.pt')
    
    # Extract and save paragraph embeddings if requested
    if hasattr(args, 'save_paragraph_embeddings') and args.save_paragraph_embeddings:
        paragraph_embeddings = extract_paragraph_embeddings(
            entity_embeddings_map, 
            method=getattr(args, 'paragraph_extraction_method', 'german_legal'),
            prefixes=getattr(args, 'paragraph_prefixes', ['§'])
        )
        
        if paragraph_embeddings:
            torch.save(paragraph_embeddings, output_dir / 'paragraph_embeddings.pt')
            
            # Save paragraph entity list
            with open(output_dir / 'paragraph_entities.txt', 'w', encoding='utf-8') as f:
                for entity in sorted(paragraph_embeddings.keys()):
                    f.write(f"{entity}\n")
            
            
            
            
            logger.info(f"Saved {len(paragraph_embeddings)} paragraph embeddings")
    
    # Save metadata as JSON
    metadata = {
        'dataset_stats': dict(dataset.stats),
        'model_config': {
            'num_entities': dataset.num_entities,
            'num_relations': dataset.num_relations,
            'embedding_dim': args.embedding_dim,
            'margin': args.margin,
            'l2_reg': args.l2_reg,
            'dropout': args.dropout,
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'scheduler': args.scheduler,
            'negative_sampling': args.negative_sampling,
        },
        'final_losses': {
            'train_loss': train_losses[-1] if train_losses else None,
            'val_loss': val_losses[-1] if val_losses else None,
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model artifacts saved to: {output_dir}")


def extract_paragraph_embeddings(entity_embeddings: Dict[str, np.ndarray], 
                               method: str = 'german_legal',
                               prefixes: List[str] = None) -> Dict[str, np.ndarray]:
    """Extract paragraph embeddings from entity embeddings."""
    if prefixes is None:
        prefixes = ['§']
    
    paragraph_embeddings = {}
    
    if method == 'german_legal':
        # German legal paragraph patterns
        patterns = [
            r'^§\s*\d+',  # § followed by number
            r'^Art\.\s*\d+',  # Art. followed by number
            r'^Abs\.\s*\d+',  # Abs. followed by number
            r'^para_\d+',  # para_ prefix
            r'^paragraph_\d+',  # paragraph_ prefix
            r'^\d+\.\d+',  # Section numbering
        ]
        
        compiled_patterns = [re.compile(pattern) for pattern in patterns]
        
        for entity, embedding in entity_embeddings.items():
            # Check prefixes
            if any(entity.startswith(prefix) for prefix in prefixes):
                paragraph_embeddings[entity] = embedding
            # Check regex patterns
            elif any(pattern.match(entity) for pattern in compiled_patterns):
                paragraph_embeddings[entity] = embedding
    
    elif method == 'prefix':
        for entity, embedding in entity_embeddings.items():
            if any(entity.startswith(prefix) for prefix in prefixes):
                paragraph_embeddings[entity] = embedding

    elif method == 'regex':
        paragraph_pattern = re.compile(args.paragraph_pattern)
        for entity, embedding in entity_embeddings.items():
            if paragraph_pattern.match(entity):
                paragraph_embeddings[entity] = embedding
            else:
                logger.warning(f"Entity does not match paragraph pattern: {entity}")
    else:
        logger.warning(f"Unknown method for extracting paragraph embeddings: {method}")

    if not paragraph_embeddings:
        logger.warning("No paragraph embeddings found with the specified method and prefixes.")
    return paragraph_embeddings


def train_transe_model(args):
    """Main function to orchestrate the training of the TransE model."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data Loading
    dataset = TripletDataset(
        filepath=args.data_path, 
        delimiter=args.delimiter,
        validation_split=args.validation_split
    )
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Model Initialization
    model = TransE(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        l2_reg=args.l2_reg,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)
    else:
        scheduler = None

    # Loss function and utilities
    criterion = nn.MarginRankingLoss(margin=args.margin, reduction='mean')
    negative_sampler = NegativeSampler(dataset.num_entities, strategy=args.negative_sampling)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    
    # Training tracking
    train_losses = []
    val_losses = []

    # Training Loop
    logger.info("--- Starting Enhanced TransE Training ---")
    
    for epoch in range(args.epochs):
        model.train()
        total_epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch in progress_bar:
            h, r, t = batch
            h, r, t = h.to(device), r.to(device), t.to(device)

            # Negative Sampling
            corrupted_h, corrupted_t = negative_sampler.sample(len(h), device, h, t)
            
            optimizer.zero_grad()

            # Calculate scores
            positive_scores = model(h, r, t)
            negative_scores_h = model(corrupted_h, r, t)
            negative_scores_t = model(h, r, corrupted_t)
            
            # Use average of head and tail corruption
            negative_scores = (negative_scores_h + negative_scores_t) / 2

            # Calculate main loss
            target = torch.tensor([-1], dtype=torch.float, device=device)
            main_loss = criterion(positive_scores, negative_scores, target)
            
            # Add regularization
            reg_loss = model.get_regularization_loss()
            total_loss = main_loss + reg_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'main': main_loss.item(),
                'reg': reg_loss.item()
            })

        avg_epoch_loss = total_epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        # Validation
        val_metrics = evaluate_model(model, dataset, device, args.batch_size)
        val_loss = val_metrics['val_loss']
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info("--- Training Complete ---")

    # Save results
    save_model_artifacts(model, dataset, args, train_losses, val_losses)
    plot_training_history(train_losses, val_losses, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an enhanced TransE model on a knowledge graph.")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to the triplet data file (e.g., 'graph_triplets.txt').")
    parser.add_argument("--delimiter", type=str, default='\t', 
                       help="Delimiter used in the triplet file.")
    parser.add_argument("--validation_split", type=float, default=0.1,
                       help="Fraction of data to use for validation.")
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=100, 
                       help="Dimension of the embeddings.")
    parser.add_argument("--margin", type=float, default=1.0, 
                       help="Margin for the MarginRankingLoss.")
    parser.add_argument("--l2_reg", type=float, default=0.01,
                       help="L2 regularization strength.")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout probability.")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, 
                       help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                       help="Learning rate for the optimizer.")
    parser.add_argument("--scheduler", type=str, choices=['plateau', 'cosine', 'none'], 
                       default='plateau', help="Learning rate scheduler.")
    parser.add_argument("--negative_sampling", type=str, choices=['uniform', 'type_aware'], 
                       default='uniform', help="Negative sampling strategy.")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                       help="Early stopping patience.")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers.")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility.")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./embeddings", 
                       help="Directory to save the trained embeddings.")
    parser.add_argument("--save_filename", type=str, default="transe_entity_embeddings.pt", 
                       help="Filename for the saved entity embeddings.")
    parser.add_argument("--save_paragraph_embeddings", action="store_true",
                       help="Extract and save paragraph embeddings separately.")
    parser.add_argument("--paragraph_extraction_method", type=str, 
                       choices=['german_legal', 'prefix','regex'], default='german_legal',
                       help="Method to identify paragraph entities.")
    parser.add_argument("--paragraph_prefixes", type=str, nargs='+', 
                       default=['§'],
                       help="Prefixes to identify paragraph entities.")
    parser.add_argument("--paragraph_pattern", type=str, default=r'§\d+', 
                       help="Regex pattern to identify paragraph entities.")

    args = parser.parse_args()
    train_transe_model(args)

    # load paragraph embeddings , log entities and embdeddings
    if args.save_paragraph_embeddings:
        output_dir = Path(args.output_dir)
        paragraph_embeddings = torch.load(output_dir / 'paragraph_embeddings.pt',weights_only=False)
        
        logger.info(f"Loaded {len(paragraph_embeddings)} paragraph embeddings.")
        
        # save entities and corresponding embeddings in dictionary format json
        paragraph_entities = {entity: embedding.tolist() for entity, embedding in paragraph_embeddings.items()}
        logger.info(f"Mapped {len(paragraph_entities)} paragraph entities to their embeddings.")
        # save dictionary to json file
        with open(output_dir / 'paragraph_embeddings_lists.json', 'w', encoding='utf-8') as f:
            json.dump(paragraph_entities, f, indent=4, ensure_ascii=False)


        # save dictionary of paragraph embeddings id to tensor (save in tensors format)
        paragraph_embeddings_tensor = {entity: torch.tensor(embedding) for entity, embedding in paragraph_entities.items()}
        logger.info(f"Converted paragraph embeddings to tensors for {len(paragraph_embeddings_tensor)} entities.")
        # save tensors to file
        torch.save(paragraph_embeddings_tensor, output_dir / 'paragraph_embeddings_tensors.json')
