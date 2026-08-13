#!/usr/bin/env python3
"""
Combine and Clean Synthetic Q&A Dataset (Optimized for Speed)

This script combines multiple JSONL files, removes duplicates, and validates data.

Optimization Notes:
- The near-duplicate check now groups samples by 'paragraph_id' first. This is a
  strong heuristic for Q&A pairs, changing the slow O(N^2) comparison across the
  entire dataset into much faster comparisons within small groups (O(sum n_i^2)).
- Exact duplicates are handled quickly via hashing before any slow string comparison.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import hashlib
from difflib import SequenceMatcher
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

REQUIRED_FIELDS = {
    'question',
    'answer',
    'label',
    'quality_score',
    'paragraph_id',
    'paragraph',
    'sentence'
}

MIN_QUALITY_SCORE = 0.5  # Minimum quality threshold
MIN_QUESTION_LENGTH = 10
MIN_ANSWER_LENGTH = 10
SIMILARITY_THRESHOLD = 0.85  # For near-duplicate detection

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_hash(question: str, answer: str) -> str:
    """Calculate hash for Q&A pair for exact duplicate detection"""
    # Use normalized text for hashing
    combined = f"{question.lower().strip()}|{answer.lower().strip()}"
    return hashlib.md5(combined.encode()).hexdigest()

def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings using SequenceMatcher"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def is_near_duplicate(qa1: Dict, qa2: Dict, similarity_threshold: float = 0.85) -> bool:
    """
    Check if two Q&A pairs are near-duplicates based on question and answer similarity.
    (Assumes exact duplicates were already removed via hash check)
    """
    q_sim = similarity_ratio(qa1['question'], qa2['question'])
    a_sim = similarity_ratio(qa1['answer'], qa2['answer'])
    
    # Near-duplicate: similar questions AND similar answers
    if q_sim >= similarity_threshold and a_sim >= similarity_threshold:
        return True
    
    return False

def validate_sample(sample: Dict) -> Tuple[bool, str]:
    """
    Validate a single Q&A sample
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    missing_fields = REQUIRED_FIELDS - set(sample.keys())
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    # Check field types and values
    if not isinstance(sample['question'], str) or len(sample['question']) < MIN_QUESTION_LENGTH:
        return False, f"Invalid question (min length: {MIN_QUESTION_LENGTH})"
    
    if not isinstance(sample['answer'], str) or len(sample['answer']) < MIN_ANSWER_LENGTH:
        return False, f"Invalid answer (min length: {MIN_ANSWER_LENGTH})"
    
    # Ensure label can be normalized before trying to normalize it
    try:
        normalize_label(sample['label'])
    except ValueError:
        return False, f"Invalid label: {sample['label']}"
    
    if not isinstance(sample['quality_score'], (int, float)):
        return False, "Invalid quality_score (must be numeric)"
    
    if float(sample['quality_score']) < MIN_QUALITY_SCORE:
        return False, f"Quality score too low: {sample['quality_score']} (min: {MIN_QUALITY_SCORE})"
    
    if not isinstance(sample['paragraph_id'], str):
        return False, "Invalid paragraph_id"
    
    if not isinstance(sample['sentence'], str):
        return False, "Invalid sentence"
    
    return True, ""

def normalize_label(label) -> str:
    """Normalize label to standard format (yes/no)"""
    if isinstance(label, bool):
        return 'yes' if label else 'no'
    
    label_str = str(label).lower().strip()
    
    if label_str in ['1', 'true', 'correct', 'yes', 'y']:
        return 'yes'
    elif label_str in ['0', 'false', 'incorrect', 'no', 'n']:
        return 'no'
    else:
        raise ValueError(f"Cannot normalize label: {label}")

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace and strip
    return ' '.join(text.split()).strip()

def normalize_sample(sample: Dict) -> Dict:
    """Normalize sample fields and calculate hash for later use"""
    normalized = {
        'question': clean_text(sample['question']),
        'answer': clean_text(sample['answer']),
        'label': normalize_label(sample['label']),
        'quality_score': float(sample['quality_score']),
        'paragraph_id': sample['paragraph_id'].strip(),
        'paragraph': clean_text(sample['paragraph']),
        'sentence': clean_text(sample['sentence']),
    }
    
    # Calculate hash once during normalization
    normalized['qa_hash'] = calculate_hash(normalized['question'], normalized['answer'])
    
    # Add optional fields if present
    for key in ['is_yes_no_question', 'language', 'generation_attempt', 'sentence_length']:
        if key in sample:
            # Simple conversion for standard types
            if key == 'is_yes_no_question':
                 normalized[key] = bool(sample[key])
            elif key in ['generation_attempt', 'sentence_length']:
                 normalized[key] = int(sample[key])
            else:
                normalized[key] = str(sample[key]).lower()
    
    if 'language' not in normalized:
        normalized['language'] = 'en'
    
    return normalized

# =============================================================================
# MAIN PROCESSING CLASS
# =============================================================================

class SyntheticQACleaner:
    """Combine, clean, and deduplicate synthetic Q&A dataset"""
    
    def __init__(self, similarity_threshold: float = 0.85, random_seed: int = 42):
        self.similarity_threshold = similarity_threshold
        self.random_seed = random_seed
        random.seed(random_seed)
        
        self.samples = []
        self.statistics = {
            'total_loaded': 0,
            'invalid_samples': 0,
            'exact_duplicates_removed': 0,
            'near_duplicates_removed': 0,
            'final_count': 0,
            'by_label': defaultdict(int),
            'by_quality': defaultdict(int),
            'by_language': defaultdict(int),
            'errors': defaultdict(int)
        }
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load samples from JSONL file"""
        samples = []
        logger.info(f"Loading from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"  Invalid JSON at line {line_num}: {e}")
                        self.statistics['errors']['json_decode'] += 1
                        continue
            
            logger.info(f"  Loaded {len(samples)} samples")
            return samples
        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
    
    def combine_files(self, file_paths: List[str]) -> None:
        """Combine multiple JSONL files"""
        logger.info(f"\n{'='*70}")
        logger.info("STEP 1: LOADING FILES")
        logger.info(f"{'='*70}")
        
        all_samples = []
        
        for file_path in file_paths:
            samples = self.load_jsonl(file_path)
            all_samples.extend(samples)
            self.statistics['total_loaded'] += len(samples)
        
        logger.info(f"Total samples loaded: {self.statistics['total_loaded']}")
        self.samples = all_samples
    
    def validate_and_clean(self) -> None:
        """Validate and clean samples"""
        logger.info(f"\n{'='*70}")
        logger.info("STEP 2: VALIDATION AND CLEANING")
        logger.info(f"{'='*70}")
        
        valid_samples = []
        
        for sample in self.samples:
            is_valid, error_msg = validate_sample(sample)
            
            if not is_valid:
                self.statistics['invalid_samples'] += 1
                self.statistics['errors']['validation'] += 1
                continue
            
            try:
                normalized = normalize_sample(sample)
                valid_samples.append(normalized)
            except Exception as e:
                self.statistics['invalid_samples'] += 1
                self.statistics['errors']['normalization'] += 1
        
        self.samples = valid_samples
        logger.info(f"Valid samples after cleaning: {len(self.samples)}")
        logger.info(f"Invalid samples removed: {self.statistics['invalid_samples']}")
    
    def remove_duplicates(self) -> None:
        """
        Optimized duplicate removal:
        1. Remove exact duplicates quickly using hashing.
        2. Group remaining samples by 'paragraph_id'.
        3. Perform the slow near-duplicate check only within these small groups.
        """
        logger.info(f"\n{'='*70}")
        logger.info("STEP 3: OPTIMIZED DUPLICATE REMOVAL")
        logger.info(f"{'='*70}")
        
        # 1. Separate into exact duplicates and unique candidates
        unique_candidates = []
        seen_hashes = set()
        
        for sample in self.samples:
            sample_hash = sample['qa_hash']
            if sample_hash in seen_hashes:
                self.statistics['exact_duplicates_removed'] += 1
                continue
            
            seen_hashes.add(sample_hash)
            unique_candidates.append(sample)
            
        logger.info(f"Samples remaining after exact deduplication: {len(unique_candidates)}")
        
        # 2. Group candidates by paragraph_id (Heuristic for near-duplicate reduction)
        samples_by_paragraph = defaultdict(list)
        for sample in unique_candidates:
            samples_by_paragraph[sample['paragraph_id']].append(sample)
            
        # 3. Perform near-duplicate check within each paragraph group
        final_unique_samples = []
        total_near_dups_removed = 0
        
        for paragraph_id, paragraph_samples in samples_by_paragraph.items():
            paragraph_uniques = []
            
            # This is the O(n_i^2) step, but n_i is small (samples per paragraph)
            for new_sample in paragraph_samples:
                is_near_dup = False
                for existing_sample in paragraph_uniques:
                    if is_near_duplicate(new_sample, existing_sample, self.similarity_threshold):
                        is_near_dup = True
                        break
                
                if not is_near_dup:
                    paragraph_uniques.append(new_sample)
                else:
                    total_near_dups_removed += 1
                    
            final_unique_samples.extend(paragraph_uniques)

        self.samples = final_unique_samples
        self.statistics['near_duplicates_removed'] = total_near_dups_removed
        logger.info(f"Unique samples after deduplication: {len(self.samples)}")
        logger.info(f"  Exact duplicates removed: {self.statistics['exact_duplicates_removed']}")
        logger.info(f"  Near-duplicates removed: {self.statistics['near_duplicates_removed']}")
    
    def balance_labels(self, target_ratio: float = 0.5) -> None:
        """Balance dataset by label (if imbalanced)"""
        logger.info(f"\n{'='*70}")
        logger.info("STEP 4: LABEL BALANCING")
        logger.info(f"{'='*70}")
        
        # Count labels
        yes_samples = [s for s in self.samples if s['label'] == 'yes']
        no_samples = [s for s in self.samples if s['label'] == 'no']
        
        if not self.samples:
            logger.warning("No samples remaining to balance.")
            return

        logger.info(f"Current label distribution:")
        logger.info(f"  Yes: {len(yes_samples)} ({len(yes_samples)/len(self.samples)*100:.1f}%)")
        logger.info(f"  No:  {len(no_samples)} ({len(no_samples)/len(self.samples)*100:.1f}%)")
        
        # NOTE: Keeping all samples as per original instruction, but warning about imbalance
        if len(yes_samples) > 0 and len(no_samples) > 0:
            ratio = min(len(yes_samples), len(no_samples)) / max(len(yes_samples), len(no_samples))
            if ratio < 0.3:  # Severe imbalance
                logger.warning(f"Dataset is severely imbalanced (ratio: {ratio:.2f})")
                logger.warning("To truly balance, undersample the majority class or collect more samples for the minority class.")
        
        logger.info("Keeping all samples (no active undersampling applied)")

        # chop to balanced size if needed
        min_size = min(len(yes_samples), len(no_samples))
        balanced_samples = yes_samples[:min_size] + no_samples[:min_size]
        self.samples = balanced_samples
        logger.info(f"Balanced dataset size: {len(self.samples)}")
    
    def shuffle(self) -> None:
        """Shuffle dataset"""
        logger.info(f"\n{'='*70}")
        logger.info("STEP 5: SHUFFLING")
        logger.info(f"{'='*70}")
        
        random.shuffle(self.samples)
        logger.info(f"Dataset shuffled with seed: {self.random_seed}")
    
    def generate_statistics(self) -> Dict:
        """Generate detailed statistics"""
        logger.info(f"\n{'='*70}")
        logger.info("STEP 6: STATISTICS")
        logger.info(f"{'='*70}")
        
        if not self.samples:
            logger.warning("No samples to generate statistics for.")
            self.statistics['final_count'] = 0
            return self.statistics

        # Label distribution
        label_counts = defaultdict(int)
        quality_distribution = defaultdict(int)
        language_counts = defaultdict(int)
        paragraph_counts = defaultdict(int)
        
        for sample in self.samples:
            label_counts[sample['label']] += 1
            quality_distribution[f"{sample['quality_score']:.1f}"] += 1
            language_counts[sample['language']] += 1
            paragraph_counts[sample['paragraph_id']] += 1
        
        self.statistics['final_count'] = len(self.samples)
        self.statistics['by_label'] = dict(label_counts)
        self.statistics['by_quality'] = dict(quality_distribution)
        self.statistics['by_language'] = dict(language_counts)
        self.statistics['unique_paragraphs'] = len(paragraph_counts)
        
        # Calculate quality metrics
        quality_scores = [s['quality_score'] for s in self.samples]
        quality_scores.sort()
        self.statistics['quality_stats'] = {
            'mean': sum(quality_scores) / len(quality_scores),
            'min': quality_scores[0],
            'max': quality_scores[-1],
            'median': quality_scores[len(quality_scores)//2]
        }
        
        # Length statistics
        question_lengths = [len(s['question'].split()) for s in self.samples]
        answer_lengths = [len(s['answer'].split()) for s in self.samples]
        
        self.statistics['length_stats'] = {
            'avg_question_words': sum(question_lengths) / len(question_lengths),
            'avg_answer_words': sum(answer_lengths) / len(answer_lengths),
            'max_question_words': max(question_lengths),
            'max_answer_words': max(answer_lengths)
        }
        
        return self.statistics
    
    def print_statistics(self) -> None:
        """Print statistics report"""
        stats = self.statistics
        
        if stats['final_count'] == 0 and stats['total_loaded'] == 0:
            print("No data processed.")
            return

        print(f"\n{'='*70}")
        print("DATA PROCESSING SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nLoading & Cleaning:")
        print(f"  Total samples loaded:        {stats['total_loaded']}")
        print(f"  Invalid samples removed:     {stats['invalid_samples']}")
        print(f"  Valid samples:               {stats['total_loaded'] - stats['invalid_samples']}")
        
        print(f"\nDeduplication:")
        print(f"  Exact duplicates removed:    {stats['exact_duplicates_removed']}")
        print(f"  Near-duplicates removed:     {stats['near_duplicates_removed']}")
        print(f"  Final unique samples:        {stats['final_count']}")
        
        if stats['final_count'] > 0:
            print(f"\nLabel Distribution:")
            for label, count in sorted(stats['by_label'].items()):
                percentage = count / stats['final_count'] * 100
                print(f"  {label.upper():10s}: {count:5d} ({percentage:5.1f}%)")
            
            print(f"\nLanguage Distribution:")
            for lang, count in sorted(stats['by_language'].items()):
                percentage = count / stats['final_count'] * 100
                print(f"  {lang.upper():10s}: {count:5d} ({percentage:5.1f}%)")
            
            print(f"\nQuality Score Distribution:")
            print(f"  Mean:     {stats['quality_stats']['mean']:.3f}")
            print(f"  Median:   {stats['quality_stats']['median']:.3f}")
            print(f"  Min:      {stats['quality_stats']['min']:.3f}")
            print(f"  Max:      {stats['quality_stats']['max']:.3f}")
            
            print(f"\nLength Statistics:")
            print(f"  Avg question length: {stats['length_stats']['avg_question_words']:.1f} words")
            print(f"  Avg answer length:   {stats['length_stats']['avg_answer_words']:.1f} words")
            print(f"  Max question length: {stats['length_stats']['max_question_words']} words")
            print(f"  Max answer length:   {stats['length_stats']['max_answer_words']} words")
            
            print(f"\nMetadata:")
            print(f"  Unique paragraphs:   {stats['unique_paragraphs']}")
        
        if stats['errors']:
            print(f"\nErrors encountered:")
            for error_type, count in stats['errors'].items():
                print(f"  {error_type:20s}: {count}")
        
        print(f"\n{'='*70}")
    
    def save_output(self, output_path: str) -> None:
        """Save cleaned and deduplicated dataset"""
        logger.info(f"\n{'='*70}")
        logger.info("STEP 7: SAVING OUTPUT")
        logger.info(f"{'='*70}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in self.samples:
                    # Remove the temporary hash field before saving
                    sample_to_save = {k: v for k, v in sample.items() if k != 'qa_hash'}
                    f.write(json.dumps(sample_to_save, ensure_ascii=False) + '\n')
            
            logger.info(f"Output saved to: {output_path}")
            logger.info(f"Total samples written: {len(self.samples)}")
        
        except Exception as e:
            logger.error(f"Error saving output: {e}")
            raise
    
    def save_statistics(self, stats_path: str) -> None:
        """Save statistics to JSON file"""
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.statistics, f, indent=2, default=str)
            
            logger.info(f"Statistics saved to: {stats_path}")
        
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")
            raise
    
    def process(self, input_files: List[str], output_file: str, 
                  stats_file: str = None) -> None:
        """Execute complete processing pipeline"""
        logger.info("\n" + "="*70)
        logger.info("SYNTHETIC Q&A DATASET PROCESSING PIPELINE")
        logger.info("="*70)
        logger.info(f"Processing {len(input_files)} input files...")
        logger.info(f"Output file: {output_file}")
        if stats_file:
            logger.info(f"Statistics file: {stats_file}")
        logger.info("="*70)
        
        # Execute pipeline
        self.combine_files(input_files)
        self.validate_and_clean()
        self.remove_duplicates()
        self.balance_labels()
        self.shuffle()
        self.generate_statistics()
        
        # Output
        self.save_output(output_file)
        if stats_file:
            self.save_statistics(stats_file)
        
        # Print summary
        self.print_statistics()
        
        logger.info(f"\n{'='*70}")
        logger.info("PROCESSING COMPLETE ✅")
        logger.info(f"{'='*70}")

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combine, clean, and deduplicate synthetic Q&A dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine and clean multiple files
  python COMBINE_AND_CLEAN_SYNTHETIC_QA_OPTIMIZED.py \\
    --input qa_chocolatine.jsonl qa_mistral_fast.jsonl qa_llama_highquality.jsonl \\
    --output qa_combined_clean.jsonl \\
    --stats qa_statistics.json
  
  # With custom quality threshold
  python COMBINE_AND_CLEAN_SYNTHETIC_QA_OPTIMIZED.py \\
    --input *.jsonl \\
    --output qa_final.jsonl \\
    --min-quality 0.6
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        required=True,
        help='Input JSONL files to combine'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '--stats', '-s',
        default=None,
        help='Output statistics JSON file (optional)'
    )
    
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.6,
        help='Minimum quality score threshold (default: 0.6)'
    )
    
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.85,
        help='Near-duplicate similarity threshold (default: 0.85)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Update configuration
    global MIN_QUALITY_SCORE, SIMILARITY_THRESHOLD
    MIN_QUALITY_SCORE = args.min_quality
    SIMILARITY_THRESHOLD = args.similarity_threshold

    # Run the processing pipeline
    cleaner = SyntheticQACleaner(
        similarity_threshold=SIMILARITY_THRESHOLD, 
        random_seed=args.seed
    )
    
    try:
        cleaner.process(args.input, args.output, args.stats)
    except Exception as e:
        logger.error(f"A fatal error occurred during processing: {e}")
        exit(1)

if __name__ == '__main__':
    main()