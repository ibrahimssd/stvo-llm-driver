#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

def validate_dataset(filepath):
    """Validate generated Q&A dataset"""
    
    stats = {
        'total_pairs': 0,
        'valid_pairs': 0,
        'invalid_pairs': 0,
        'avg_quality_score': 0,
        'label_distribution': defaultdict(int),
        'quality_score_distribution': defaultdict(int),
        'issues': []
    }
    
    quality_scores = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    pair = json.loads(line)
                    stats['total_pairs'] += 1
                    
                    # Validate structure
                    if all(k in pair for k in ['question', 'answer', 'label']):
                        stats['valid_pairs'] += 1
                        
                        # Collect metrics
                        stats['label_distribution'][pair['label']] += 1
                        
                        if 'quality_score' in pair:
                            quality_scores.append(pair['quality_score'])
                            score_bucket = int(pair['quality_score'] * 10) / 10
                            stats['quality_score_distribution'][score_bucket] += 1
                    else:
                        stats['invalid_pairs'] += 1
                        stats['issues'].append(f"Line {line_num}: Missing required fields")
                
                except json.JSONDecodeError as e:
                    stats['invalid_pairs'] += 1
                    stats['issues'].append(f"Line {line_num}: Invalid JSON - {str(e)}")
        
        if quality_scores:
            stats['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
        
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    
    return stats

def print_validation_report(stats):
    """Print validation report"""
    print("\n" + "="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Valid pairs: {stats['valid_pairs']}")
    print(f"Invalid pairs: {stats['invalid_pairs']}")
    print(f"Average quality score: {stats['avg_quality_score']:.3f}")
    print(f"\nLabel distribution: {dict(stats['label_distribution'])}")
    print(f"\nQuality score distribution:")
    for score in sorted(stats['quality_score_distribution'].keys()):
        count = stats['quality_score_distribution'][score]
        percentage = (count / stats['valid_pairs'] * 100) if stats['valid_pairs'] > 0 else 0
        print(f"  {score:.1f}-{score+0.1:.1f}: {count} ({percentage:.1f}%)")
    
    if stats['issues']:
        print(f"\nIssues found (first 10):")
        for issue in stats['issues'][:10]:
            print(f"  - {issue}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_qa_dataset.py <dataset_path>")
        sys.exit(1)
    
    stats = validate_dataset(sys.argv[1])
    if stats:
        print_validation_report(stats)
