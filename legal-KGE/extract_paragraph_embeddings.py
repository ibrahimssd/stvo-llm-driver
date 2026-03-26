"""
Extract paragraph embeddings from trained KGE models
Usage: python extract_paragraph_embeddings.py --checkpoint_path ./embeddings/checkpoint_complex_final.pt --output_dir ./paragraph_embeddings
"""

import torch
import json
import argparse
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_paragraph_embeddings(entity_embeddings_map, method='german_legal'):
    """
    Extract paragraph embeddings from entity embeddings.
    
    Args:
        entity_embeddings_map: Dictionary mapping entity names to embeddings (numpy arrays or tensors)
        method: Extraction method ('german_legal', 'prefix', or 'all_numeric')
    
    Returns:
        Dictionary of paragraph embeddings
    """
    paragraph_embeddings = {}
    
    if method == 'german_legal':
        # Comprehensive German legal paragraph patterns
        patterns = [
            r'^§\s*\d+',                    # § 52, § 21, etc.
            r'^§\s*\d+\s+Abs',              # § 52 Abs, § 21 Abs
            r'^§\s*\d+\s+Absatz',           # § 52 Absatz
            r'^Art\.\s*\d+',                # Art. 1, Art. 20
            r'^Artikel\s+\d+',              # Artikel 1
            r'^Abs\.\s*\d+',                # Abs. 1, Abs. 2
            r'^Absatz\s+\d+',               # Absatz 1
            r'^para_\d+',                   # para_52, para_21
            r'^paragraph_\d+',              # paragraph_52
            r'^Paragraph\s+\d+',            # Paragraph 52
            r'^\d+\.\d+',                   # 52.1, 21.3
            r'^StVO_§\d+',                  # StVO_§52
            r'^§\d+StVO',                   # §52StVO
        ]
        
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        for entity, embedding in entity_embeddings_map.items():
            if any(pattern.match(entity) for pattern in compiled_patterns):
                paragraph_embeddings[entity] = embedding
                
    elif method == 'prefix':
        # Simple prefix matching
        prefixes = ['§', 'Art.', 'Artikel', 'Abs.', 'Absatz', 'para_', 'paragraph_', 'Paragraph']
        for entity, embedding in entity_embeddings_map.items():
            if any(entity.startswith(prefix) for prefix in prefixes):
                paragraph_embeddings[entity] = embedding
                
    elif method == 'all_numeric':
        # Match any entity that starts with a number or §
        numeric_pattern = re.compile(r'^[\d§]')
        for entity, embedding in entity_embeddings_map.items():
            if numeric_pattern.match(entity):
                paragraph_embeddings[entity] = embedding
    
    return paragraph_embeddings


def load_and_extract(checkpoint_path, output_dir, method='german_legal'):
    """
    Load checkpoint and extract paragraph embeddings.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for paragraph embeddings
        method: Extraction method
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model_type = checkpoint.get('model_type', 'unknown')
    entity_to_id = checkpoint.get('entity_to_id', {})
    
    logger.info(f"Model type: {model_type}")
    logger.info(f"Total entities: {len(entity_to_id)}")
    
    # Try to load entity embeddings from checkpoint directory
    checkpoint_dir = Path(checkpoint_path).parent
    
    # Look for entity embeddings file
    entity_emb_files = list(checkpoint_dir.glob(f"{model_type}_entity_embeddings_*.pt"))
    
    if not entity_emb_files:
        logger.error(f"No entity embeddings file found in {checkpoint_dir}")
        logger.info(f"Looking for: {model_type}_entity_embeddings_*.pt")
        return
    
    # Use the most recent one
    entity_emb_file = entity_emb_files[-1]
    logger.info(f"Loading entity embeddings from: {entity_emb_file}")
    
    entity_embeddings_map = torch.load(entity_emb_file, map_location='cpu', weights_only=False)
    
    logger.info(f"Loaded {len(entity_embeddings_map)} entity embeddings")
    
    # Extract paragraph embeddings
    logger.info(f"Extracting paragraph embeddings using method: {method}")
    paragraph_embeddings = extract_paragraph_embeddings(entity_embeddings_map, method=method)
    
    if not paragraph_embeddings:
        logger.warning("No paragraph embeddings found!")
        logger.info("Sample entities:")
        for i, entity in enumerate(list(entity_embeddings_map.keys())[:10]):
            logger.info(f"  {i+1}. {entity}")
        return
    
    logger.info(f"Extracted {len(paragraph_embeddings)} paragraph embeddings")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to appropriate format
    import numpy as np
    
    # Save as PyTorch tensors
    para_emb_tensors = {}
    para_emb_lists = {}
    
    for entity, emb in paragraph_embeddings.items():
        if isinstance(emb, np.ndarray):
            para_emb_tensors[entity] = torch.from_numpy(emb)
            para_emb_lists[entity] = emb.tolist()
        elif isinstance(emb, torch.Tensor):
            para_emb_tensors[entity] = emb
            para_emb_lists[entity] = emb.numpy().tolist()
        else:
            logger.warning(f"Unknown embedding type for {entity}: {type(emb)}")
    
    # Save in multiple formats
    
    # 1. PyTorch tensors
    tensor_file = output_dir / f'{model_type}_paragraph_embeddings.pt'
    torch.save(para_emb_tensors, tensor_file)
    logger.info(f"Saved PyTorch tensors to: {tensor_file}")
    
    # 2. JSON lists
    json_file = output_dir / f'{model_type}_paragraph_embeddings.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(para_emb_lists, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON lists to: {json_file}")
    
    # 3. Entity list
    list_file = output_dir / f'{model_type}_paragraph_entities.txt'
    with open(list_file, 'w', encoding='utf-8') as f:
        for entity in sorted(paragraph_embeddings.keys()):
            f.write(f"{entity}\n")
    logger.info(f"Saved entity list to: {list_file}")
    
    # 4. Summary statistics
    stats_file = output_dir / f'{model_type}_paragraph_stats.json'
    
    # Get embedding dimension
    first_emb = next(iter(paragraph_embeddings.values()))
    if isinstance(first_emb, np.ndarray):
        emb_dim = first_emb.shape[0]
    else:
        emb_dim = first_emb.size(0)
    
    stats = {
        'model_type': model_type,
        'total_entities': len(entity_embeddings_map),
        'paragraph_entities': len(paragraph_embeddings),
        'extraction_method': method,
        'embedding_dimension': int(emb_dim),
        'sample_paragraphs': sorted(list(paragraph_embeddings.keys()))[:10]
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved statistics to: {stats_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXTRACTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total entities in model: {len(entity_embeddings_map)}")
    logger.info(f"Paragraph entities extracted: {len(paragraph_embeddings)}")
    logger.info(f"Extraction rate: {len(paragraph_embeddings)/len(entity_embeddings_map)*100:.1f}%")
    logger.info(f"Embedding dimension: {emb_dim}")
    logger.info(f"\nSample paragraphs:")
    for i, entity in enumerate(sorted(paragraph_embeddings.keys())[:10]):
        logger.info(f"  {i+1}. {entity}")
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract paragraph embeddings from trained KGE models")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for paragraph embeddings")
    parser.add_argument("--method", type=str, default='german_legal',
                       choices=['german_legal', 'prefix', 'all_numeric'],
                       help="Extraction method for identifying paragraphs")
    
    args = parser.parse_args()
    
    load_and_extract(args.checkpoint_path, args.output_dir, args.method)


if __name__ == "__main__":
    main()