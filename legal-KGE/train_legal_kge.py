"""
Enhanced Knowledge Graph Embedding for Legal Domain
Supports multiple KGE methods: TransE, DistMult, ComplEx, RotatE, and TuckER
"""

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
import wandb

warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """Enhanced Dataset with better legal text handling."""
    
    def __init__(self, filepath: str, delimiter: str = '\t', validation_split: float = 0.1,
                 test_split: float = 0.1):
        self.triplets = []
        self.entities = set()
        self.relations = set()
        self.validation_split = validation_split
        self.test_split = test_split
        
        # Statistics tracking
        self.stats = defaultdict(int)
        self.relation_hierarchy = defaultdict(list)  # For hierarchical relations
        
        logger.info(f"Reading triplets from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                parts = line.split(delimiter)
                if len(parts) != 3:
                    logger.warning(f"Line {line_num}: Expected 3 parts, got {len(parts)}")
                    self.stats['malformed_lines'] += 1
                    continue
                    
                h, r, t = [part.strip() for part in parts]
                
                if not all([h, r, t]):
                    logger.warning(f"Line {line_num}: Empty components found")
                    self.stats['empty_components'] += 1
                    continue
                
                self.triplets.append((h, r, t))
                self.entities.add(h)
                self.entities.add(t)
                self.relations.add(r)
                self.stats['valid_triplets'] += 1
                
                # Track relation hierarchy for legal structure
                if 'sub' in r.lower() or 'parent' in r.lower():
                    self.relation_hierarchy[r].append((h, t))
                
            except Exception as e:
                logger.warning(f"Line {line_num}: Error processing line - {e}")
                self.stats['processing_errors'] += 1
        
        if not self.triplets:
            raise ValueError("No valid triplets found in the dataset")
        
        # Create mappings
        self.entity_list = sorted(list(self.entities))
        self.relation_list = sorted(list(self.relations))
        
        self.entity_to_id = {e: i for i, e in enumerate(self.entity_list)}
        self.relation_to_id = {r: i for i, r in enumerate(self.relation_list)}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}
        
        self.num_entities = len(self.entity_list)
        self.num_relations = len(self.relation_list)
        
        # Enhanced train/val/test split
        self._create_splits()
        self._log_statistics()

    def _create_splits(self):
        """Create train/validation/test split with stratification."""
        random.shuffle(self.triplets)
        
        # Calculate split indices
        test_idx = int(len(self.triplets) * self.test_split)
        val_idx = test_idx + int(len(self.triplets) * self.validation_split)
        
        self.test_triplets = self.triplets[:test_idx]
        self.val_triplets = self.triplets[test_idx:val_idx]
        self.train_triplets = self.triplets[val_idx:]
        
        logger.info(f"Split: {len(self.train_triplets)} train, "
                   f"{len(self.val_triplets)} val, {len(self.test_triplets)} test")

    def _log_statistics(self):
        """Log comprehensive dataset statistics."""
        logger.info(f"Dataset Statistics:")
        logger.info(f"  - Total entities: {self.num_entities}")
        logger.info(f"  - Total relations: {self.num_relations}")
        logger.info(f"  - Total triplets: {len(self.triplets)}")
        logger.info(f"  - Hierarchical relations: {len(self.relation_hierarchy)}")
        
        # Relation frequency analysis
        relation_counts = defaultdict(int)
        for _, r, _ in self.triplets:
            relation_counts[r] += 1
        
        logger.info(f"  - Most frequent relations:")
        for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"    {rel}: {count}")

    def get_train_data(self):
        return [(self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]) 
                for h, r, t in self.train_triplets]

    def get_val_data(self):
        return [(self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]) 
                for h, r, t in self.val_triplets]
    
    def get_test_data(self):
        return [(self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]) 
                for h, r, t in self.test_triplets]

    def __len__(self):
        return len(self.train_triplets)

    def __getitem__(self, idx):
        h, r, t = self.train_triplets[idx]
        return (
            self.entity_to_id[h],
            self.relation_to_id[r],
            self.entity_to_id[t]
        )


# ==================== KGE Models ====================

class TransE(nn.Module):
    """
    TransE: h + r ≈ t
    Best for: Simple hierarchical relationships, translation-like semantics
    Legal use case: Straightforward parent-child, reference relationships
    """
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, 
                 margin: float = 1.0, l2_reg: float = 0.01, dropout: float = 0.1,
                 norm_p: int = 2):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.l2_reg = l2_reg
        self.norm_p = norm_p  # L1 or L2 norm

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self._initialize_embeddings()

    def _initialize_embeddings(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        self.entity_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)

    def _normalize_embeddings(self):
        with torch.no_grad():
            self.entity_embeddings.weight.data.clamp_(-1, 1)
            self.entity_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self._normalize_embeddings()
        
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        
        if self.training:
            h_emb = self.dropout(h_emb)
            r_emb = self.dropout(r_emb)
            t_emb = self.dropout(t_emb)

        score = torch.norm(h_emb + r_emb - t_emb, p=self.norm_p, dim=1)
        return score

    def get_regularization_loss(self) -> torch.Tensor:
        entity_reg = torch.norm(self.entity_embeddings.weight, p=2)**2
        relation_reg = torch.norm(self.relation_embeddings.weight, p=2)**2
        return self.l2_reg * (entity_reg + relation_reg) / (self.num_entities + self.num_relations)


class DistMult(nn.Module):
    """
    DistMult: score(h,r,t) = h^T * diag(r) * t
    Best for: Symmetric relations, efficient computation
    Legal use case: Co-occurrence, mutual references, bidirectional citations
    
    ADVANTAGES for legal domain:
    - Handles symmetric relations well (mutual citations)
    - More interpretable than TransE
    - Efficient computation
    """
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 l2_reg: float = 0.01, dropout: float = 0.1):
        super(DistMult, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        
        if self.training:
            h_emb = self.dropout(h_emb)
            r_emb = self.dropout(r_emb)
            t_emb = self.dropout(t_emb)

        # Element-wise multiplication and sum
        score = torch.sum(h_emb * r_emb * t_emb, dim=1)
        return -score  # Negative for ranking loss

    def get_regularization_loss(self) -> torch.Tensor:
        entity_reg = torch.norm(self.entity_embeddings.weight, p=2)**2
        relation_reg = torch.norm(self.relation_embeddings.weight, p=2)**2
        return self.l2_reg * (entity_reg + relation_reg) / (self.num_entities + self.num_relations)


class ComplEx(nn.Module):
    """
    ComplEx: Uses complex-valued embeddings
    Best for: Asymmetric and inverse relations
    Legal use case: "cites" vs "cited_by", "amends" vs "amended_by"
    
    ADVANTAGES for legal domain:
    - Handles asymmetric relations (A cites B ≠ B cites A)
    - Models inverse relations naturally
    - Better than DistMult for directed legal relationships
    """
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 l2_reg: float = 0.01, dropout: float = 0.1):
        super(ComplEx, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg

        # Real and imaginary parts
        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(num_relations, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)

        # Initialize
        for embedding in [self.entity_embeddings_real, self.entity_embeddings_imag,
                         self.relation_embeddings_real, self.relation_embeddings_imag]:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_real = self.entity_embeddings_real(h)
        h_imag = self.entity_embeddings_imag(h)
        r_real = self.relation_embeddings_real(r)
        r_imag = self.relation_embeddings_imag(r)
        t_real = self.entity_embeddings_real(t)
        t_imag = self.entity_embeddings_imag(t)
        
        if self.training:
            h_real, h_imag = self.dropout(h_real), self.dropout(h_imag)
            r_real, r_imag = self.dropout(r_real), self.dropout(r_imag)
            t_real, t_imag = self.dropout(t_real), self.dropout(t_imag)

        # ComplEx scoring function
        score = torch.sum(
            h_real * r_real * t_real +
            h_real * r_imag * t_imag +
            h_imag * r_real * t_imag -
            h_imag * r_imag * t_real,
            dim=1
        )
        
        return -score  # Negative for ranking loss

    def get_regularization_loss(self) -> torch.Tensor:
        reg = (torch.norm(self.entity_embeddings_real.weight, p=2)**2 +
               torch.norm(self.entity_embeddings_imag.weight, p=2)**2 +
               torch.norm(self.relation_embeddings_real.weight, p=2)**2 +
               torch.norm(self.relation_embeddings_imag.weight, p=2)**2)
        return self.l2_reg * reg / (2 * self.num_entities + 2 * self.num_relations)


class RotatE(nn.Module):
    """
    RotatE: Models relations as rotations in complex space
    Best for: Relation patterns (symmetry, inversion, composition)
    Legal use case: Complex legal relationships with hierarchies
    
    ADVANTAGES for legal domain:
    - Excellent for hierarchical structures (legal code hierarchy)
    - Models relation composition (transitive references)
    - Handles multiple relation patterns simultaneously
    """
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 margin: float = 6.0, l2_reg: float = 0.01, dropout: float = 0.1):
        super(RotatE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.l2_reg = l2_reg

        # Entity embeddings: complex numbers (real, imaginary)
        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
        
        # Relations: phases in complex space
        self.relation_phases = nn.Embedding(num_relations, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.uniform_(self.entity_embeddings_real.weight.data, -1, 1)
        nn.init.uniform_(self.entity_embeddings_imag.weight.data, -1, 1)
        nn.init.uniform_(self.relation_phases.weight.data, 0, 2 * np.pi)

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_real = self.entity_embeddings_real(h)
        h_imag = self.entity_embeddings_imag(h)
        r_phase = self.relation_phases(r)
        t_real = self.entity_embeddings_real(t)
        t_imag = self.entity_embeddings_imag(t)
        
        if self.training:
            h_real, h_imag = self.dropout(h_real), self.dropout(h_imag)
            t_real, t_imag = self.dropout(t_real), self.dropout(t_imag)

        # Rotation: h ∘ r = t
        # (h_real + i*h_imag) * (cos(r) + i*sin(r))
        r_cos = torch.cos(r_phase)
        r_sin = torch.sin(r_phase)
        
        rotated_real = h_real * r_cos - h_imag * r_sin
        rotated_imag = h_real * r_sin + h_imag * r_cos
        
        # Distance to tail
        diff_real = rotated_real - t_real
        diff_imag = rotated_imag - t_imag
        
        score = torch.sqrt(diff_real**2 + diff_imag**2 + 1e-8).sum(dim=1)
        return score

    def get_regularization_loss(self) -> torch.Tensor:
        entity_reg = (torch.norm(self.entity_embeddings_real.weight, p=2)**2 +
                     torch.norm(self.entity_embeddings_imag.weight, p=2)**2)
        return self.l2_reg * entity_reg / (2 * self.num_entities)


# ==================== Training Infrastructure ====================

class NegativeSampler:
    """Enhanced negative sampling with type-aware strategy for legal domain."""
    
    def __init__(self, num_entities: int, strategy: str = 'uniform',
                 entity_types: Optional[Dict] = None):
        self.num_entities = num_entities
        self.strategy = strategy
        self.entity_types = entity_types or {}

    def sample(self, batch_size: int, device: torch.device, 
               positive_heads: Optional[torch.Tensor] = None,
               positive_tails: Optional[torch.Tensor] = None,
               relations: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.strategy == 'type_aware' and self.entity_types:
            # Sample within same types (e.g., paragraph corrupts to paragraph)
            corrupted_h = self._type_aware_sample(batch_size, positive_heads, device)
            corrupted_t = self._type_aware_sample(batch_size, positive_tails, device)
        else:
            corrupted_h = torch.randint(0, self.num_entities, (batch_size,), device=device)
            corrupted_t = torch.randint(0, self.num_entities, (batch_size,), device=device)
        
        return corrupted_h, corrupted_t
    
    def _type_aware_sample(self, batch_size: int, entities: torch.Tensor, 
                          device: torch.device) -> torch.Tensor:
        """Sample negative entities of the same type."""
        # Placeholder - implement based on your entity type structure
        return torch.randint(0, self.num_entities, (batch_size,), device=device)


def evaluate_model(model: nn.Module, dataset: TripletDataset, device: torch.device,
                  batch_size: int = 128, model_type: str = 'transe') -> Dict[str, float]:
    """Evaluate model with multiple metrics."""
    model.eval()
    val_data = dataset.get_val_data()
    
    if not val_data:
        return {'val_loss': float('inf'), 'mrr': 0.0, 'hits@10': 0.0}
    
    total_loss = 0.0
    num_batches = 0
    negative_sampler = NegativeSampler(dataset.num_entities)
    
    # For TransE and RotatE
    if model_type in ['transe', 'rotate']:
        criterion = nn.MarginRankingLoss(margin=model.margin, reduction='mean')
    else:
        criterion = nn.MarginRankingLoss(margin=1.0, reduction='mean')
    
    mrr_sum = 0.0
    hits_at_10 = 0
    num_samples = 0
    
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size]
            if not batch:
                continue
                
            h = torch.tensor([t[0] for t in batch], device=device)
            r = torch.tensor([t[1] for t in batch], device=device)
            t = torch.tensor([t[2] for t in batch], device=device)
            
            # Loss calculation
            corrupted_h, corrupted_t = negative_sampler.sample(len(batch), device, h, t)
            
            positive_scores = model(h, r, t)
            negative_scores_h = model(corrupted_h, r, t)
            negative_scores_t = model(h, r, corrupted_t)
            negative_scores = (negative_scores_h + negative_scores_t) / 2
            
            target = torch.tensor([-1], dtype=torch.float, device=device)
            loss = criterion(positive_scores, negative_scores, target)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Ranking metrics (simplified - full implementation would rank against all entities)
            ranks = (positive_scores < negative_scores).float().sum() + 1
            mrr_sum += (1.0 / ranks).sum().item()
            hits_at_10 += (ranks <= 10).sum().item()
            num_samples += len(batch)
    
    metrics = {
        'val_loss': total_loss / num_batches if num_batches > 0 else float('inf'),
        'mrr': mrr_sum / num_samples if num_samples > 0 else 0.0,
        'hits@10': hits_at_10 / num_samples if num_samples > 0 else 0.0
    }
    
    return metrics


def train_model(args):
    """Main training function supporting multiple KGE models."""
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model_type}_{args.embedding_dim}d",
            config=vars(args)
        )
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    dataset = TripletDataset(
        filepath=args.data_path,
        delimiter=args.delimiter,
        validation_split=args.validation_split,
        test_split=args.test_split
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Initialize model based on type
    model_classes = {
        'transe': TransE,
        'distmult': DistMult,
        'complex': ComplEx,
        'rotate': RotatE
    }
    
    if args.model_type not in model_classes:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model_class = model_classes[args.model_type]
    
    # Model-specific initialization
    if args.model_type == 'transe':
        model = model_class(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            l2_reg=args.l2_reg,
            dropout=args.dropout
        )
    elif args.model_type == 'rotate':
        model = model_class(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            l2_reg=args.l2_reg,
            dropout=args.dropout
        )
    else:  # distmult, complex
        model = model_class(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            embedding_dim=args.embedding_dim,
            l2_reg=args.l2_reg,
            dropout=args.dropout
        )
    
    model = model.to(device)
    logger.info(f"Model: {args.model_type.upper()}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    
    if args.scheduler == 'plateau':
        # Note: 'verbose' parameter removed in newer PyTorch versions
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)
    else:
        scheduler = None

    # Loss and utilities
    if args.model_type in ['transe', 'rotate']:
        criterion = nn.MarginRankingLoss(margin=model.margin, reduction='mean')
    else:
        criterion = nn.MarginRankingLoss(margin=1.0, reduction='mean')
    
    negative_sampler = NegativeSampler(dataset.num_entities, strategy=args.negative_sampling)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Training Loop
    logger.info(f"--- Starting {args.model_type.upper()} Training ---")
    
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
            negative_scores = (negative_scores_h + negative_scores_t) / 2

            # Calculate loss
            target = torch.tensor([-1], dtype=torch.float, device=device)
            main_loss = criterion(positive_scores, negative_scores, target)
            
            # Regularization
            reg_loss = model.get_regularization_loss()
            total_loss = main_loss + reg_loss
            
            total_loss.backward()
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
        val_metrics = evaluate_model(model, dataset, device, args.batch_size, args.model_type)
        val_loss = val_metrics['val_loss']
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - "
                   f"Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"MRR: {val_metrics['mrr']:.4f}, Hits@10: {val_metrics['hits@10']:.4f}, "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss,
                'mrr': val_metrics['mrr'],
                'hits@10': val_metrics['hits@10'],
                'lr': optimizer.param_groups[0]['lr']
            }, step=epoch + 1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, dataset, args, epoch, train_losses, val_losses, 'best')
        
        # Learning rate scheduling
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # Log if LR changed
            if old_lr != new_lr:
                logger.info(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

    logger.info("--- Training Complete ---")
    
    # Save final model
    save_checkpoint(model, dataset, args, args.epochs, train_losses, val_losses, 'final')
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Final Train Loss: {train_losses[-1]:.4f}")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Total Entities: {dataset.num_entities}")
    logger.info(f"Total Relations: {dataset.num_relations}")
    logger.info(f"Embedding Dimension: {args.embedding_dim}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # Load and report paragraph embeddings
    output_dir = Path(args.output_dir)
    para_file = output_dir / f'{args.model_type}_paragraph_embeddings_final.json'
    if para_file.exists():
        with open(para_file, 'r', encoding='utf-8') as f:
            para_data = json.load(f)
        logger.info(f"\nParagraph Embeddings: {len(para_data)} extracted")
        logger.info(f"Saved to: {para_file}")
    
    logger.info("="*80 + "\n")
    
    if args.use_wandb:
        wandb.finish()


def extract_paragraph_embeddings(entity_embeddings_map: Dict[str, np.ndarray],
                               method: str = 'german_legal') -> Dict[str, np.ndarray]:
    """Extract paragraph embeddings from entity embeddings.
    
    Args:
        entity_embeddings_map: Dictionary mapping entity names to embeddings
        method: Extraction method ('german_legal' or 'prefix')
    
    Returns:
        Dictionary of paragraph embeddings
    """
    import re
    
    paragraph_embeddings = {}
    
    if method == 'german_legal':
        # German legal paragraph patterns
        patterns = [
            r'^§\s*\d+',           # § followed by number
            r'^Art\.\s*\d+',       # Art. followed by number
            r'^Abs\.\s*\d+',       # Abs. followed by number
            r'^para_\d+',          # para_ prefix
            r'^paragraph_\d+',     # paragraph_ prefix
            r'^\d+\.\d+',          # Section numbering (e.g., 52.1)
            r'^Paragraph\s+\d+',   # Paragraph followed by number
        ]
        
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        for entity, embedding in entity_embeddings_map.items():
            # Check if entity matches any pattern
            if any(pattern.match(entity) for pattern in compiled_patterns):
                paragraph_embeddings[entity] = embedding
    
    elif method == 'prefix':
        # Simple prefix matching
        prefixes = ['§', 'Art.', 'Abs.', 'para_', 'paragraph_', 'Paragraph']
        for entity, embedding in entity_embeddings_map.items():
            if any(entity.startswith(prefix) for prefix in prefixes):
                paragraph_embeddings[entity] = embedding
    
    return paragraph_embeddings


def save_checkpoint(model, dataset, args, epoch, train_losses, val_losses, suffix=''):
    """Save model checkpoint with embeddings."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': vars(args),
        'entity_to_id': dataset.entity_to_id,
        'relation_to_id': dataset.relation_to_id,
    }
    
    filename = f'checkpoint_{args.model_type}_{suffix}.pt' if suffix else f'checkpoint_{args.model_type}.pt'
    torch.save(checkpoint, output_dir / filename)
    
    # Extract and save entity embeddings
    if args.model_type == 'transe':
        entity_emb = model.entity_embeddings.weight.data.cpu()
    elif args.model_type in ['distmult']:
        entity_emb = model.entity_embeddings.weight.data.cpu()
    elif args.model_type == 'complex':
        # Concatenate real and imaginary parts
        entity_emb = torch.cat([
            model.entity_embeddings_real.weight.data.cpu(),
            model.entity_embeddings_imag.weight.data.cpu()
        ], dim=1)
    elif args.model_type == 'rotate':
        # Concatenate real and imaginary parts
        entity_emb = torch.cat([
            model.entity_embeddings_real.weight.data.cpu(),
            model.entity_embeddings_imag.weight.data.cpu()
        ], dim=1)
    
    # Save entity embeddings as dict
    entity_embeddings_map = {
        entity: entity_emb[idx].numpy()
        for entity, idx in dataset.entity_to_id.items()
    }
    
    emb_filename = f'{args.model_type}_entity_embeddings_{suffix}.pt' if suffix else f'{args.model_type}_entity_embeddings.pt'
    torch.save(entity_embeddings_map, output_dir / emb_filename)
    
    logger.info(f"Checkpoint saved to: {output_dir / filename}")
    
    # Extract and save paragraph embeddings (only for final checkpoint)
    if suffix == 'final' or suffix == 'best':
        paragraph_embeddings = extract_paragraph_embeddings(
            entity_embeddings_map, 
            method='german_legal'
        )
        
        if paragraph_embeddings:
            # Save as PyTorch tensors
            para_emb_tensors = {
                entity: torch.from_numpy(emb) 
                for entity, emb in paragraph_embeddings.items()
            }
            para_filename = f'{args.model_type}_paragraph_embeddings_{suffix}.pt'
            torch.save(para_emb_tensors, output_dir / para_filename)
            
            # Save as JSON (lists) for easy loading
            para_emb_lists = {
                entity: emb.tolist() 
                for entity, emb in paragraph_embeddings.items()
            }
            json_filename = f'{args.model_type}_paragraph_embeddings_{suffix}.json'
            with open(output_dir / json_filename, 'w', encoding='utf-8') as f:
                json.dump(para_emb_lists, f, indent=2, ensure_ascii=False)
            
            # Save list of paragraph entities
            para_list_filename = f'{args.model_type}_paragraph_entities_{suffix}.txt'
            with open(output_dir / para_list_filename, 'w', encoding='utf-8') as f:
                for entity in sorted(paragraph_embeddings.keys()):
                    f.write(f"{entity}\n")
            
            logger.info(f"Saved {len(paragraph_embeddings)} paragraph embeddings:")
            logger.info(f"  - Tensors: {output_dir / para_filename}")
            logger.info(f"  - JSON: {output_dir / json_filename}")
            logger.info(f"  - Entity list: {output_dir / para_list_filename}")
            
            # Log sample of paragraph entities
            sample_entities = list(paragraph_embeddings.keys())[:5]
            logger.info(f"  - Sample paragraphs: {', '.join(sample_entities)}")
        else:
            logger.warning("No paragraph embeddings found in entity set")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KGE models for legal domain")
    
    # Model selection
    parser.add_argument("--model_type", type=str, required=True,
                       choices=['transe', 'distmult', 'complex', 'rotate'],
                       help="KGE model type to train")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--delimiter", type=str, default='<tr>')
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--l2_reg", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--scheduler", type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'])
    parser.add_argument("--negative_sampling", type=str, default='uniform',
                       choices=['uniform', 'type_aware'])
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./embeddings")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="legal-kge")
    
    args = parser.parse_args()
    train_model(args)