"""
Evaluation Suite for Citation-Based Legal Q&A Dataset
=====================================================
Comprehensive evaluation scripts for paper experiments
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from dataclasses import dataclass
import random
from scipy import stats
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import argparse
from llm_evaluator import BaselineLLMEvaluator




# ============= Configuration =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


# ============= Dataset Analysis =============

class DatasetAnalyzer:
    """Comprehensive dataset analysis for paper reporting"""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.data = self.load_dataset()
        self.logger = logging.getLogger(__name__)
        
   
    def load_dataset(self) -> List[Dict]:
        """
        Load from QA dataset, safely handling JSON decoding errors for each line.
        """
        data = []
        
        # Check if the path is valid before trying to open
        if not os.path.exists(self.dataset_path):
            logging.error(f"Dataset file not found at: {self.dataset_path}")
            return data

        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                # We use enumerate to get the line number for better error reporting
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue # Skip empty lines

                    try:
                        # CRITICAL FIX: The try/except block must be around the 
                        # specific operation that can fail (json.loads)
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Log the error with the line number and a snippet of the problematic data
                        logging.error(
                            f"JSON Decode Error on line {i + 1} of dataset file: {e}. "
                            f"Skipping malformed line starting with: '{line[:80]}...'"
                        )
                        # The loop continues to the next line after logging the error
                    except Exception as e:
                        logging.error(f"An unexpected error occurred processing line {i + 1}: {e}")

        except Exception as e:
            # Handles errors opening or reading the file itself
            logging.critical(f"A file reading error occurred during dataset loading: {e}")
            return data

        logging.info(f"Successfully loaded {len(data)} valid samples.")
        return data
    


    def compute_basic_statistics(self) -> Dict[str, Any]:
        """Compute basic dataset statistics"""
        
        stats = {
            'total_samples': len(self.data),
            'label_distribution': Counter([d['label'] for d in self.data]),
        }
        
        # Text length statistics
        q_lengths = [len(d['question'].split()) for d in self.data]
        a_lengths = [len(d['answer'].split()) for d in self.data]
        
        stats['question_length'] = {
            'mean': np.mean(q_lengths),
            'std': np.std(q_lengths),
            'min': np.min(q_lengths),
            'max': np.max(q_lengths),
            'median': np.median(q_lengths)
        }
        
        stats['answer_length'] = {
            'mean': np.mean(a_lengths),
            'std': np.std(a_lengths),
            'min': np.min(a_lengths),
            'max': np.max(a_lengths),
            'median': np.median(a_lengths)
        }
        
        # Paragraph coverage - now with normalized IDs
        paragraph_ids = [d['paragraph_id'] for d in self.data]
        unique_paragraphs = set(paragraph_ids)
        stats['unique_paragraphs'] = len(unique_paragraphs)
        stats['paragraph_distribution'] = Counter(paragraph_ids)
        
        # Log sample paragraphs to verify normalization
        sample_paragraphs = sorted(list(unique_paragraphs))[:5]
        self.logger.info(f"Sample normalized paragraph IDs: {sample_paragraphs}")
        
        return stats
    
    def compute_linguistic_diversity(self) -> Dict[str, float]:
        """Compute linguistic diversity metrics"""
        all_questions = ' '.join([d['question'] for d in self.data])
        all_answers = ' '.join([d['answer'] for d in self.data])
        
        # Tokenize
        q_tokens = all_questions.lower().split()
        a_tokens = all_answers.lower().split()
        
        # Unique n-grams
        def get_ngrams(tokens, n):
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        metrics = {
            'questions': {
                'unique_unigrams': len(set(q_tokens)),
                'unique_bigrams': len(set(get_ngrams(q_tokens, 2))),
                'unique_trigrams': len(set(get_ngrams(q_tokens, 3))),
                'type_token_ratio': len(set(q_tokens)) / len(q_tokens),
                'avg_sentence_length': np.mean([len(d['question'].split()) for d in self.data])
            },
            'answers': {
                'unique_unigrams': len(set(a_tokens)),
                'unique_bigrams': len(set(get_ngrams(a_tokens, 2))),
                'unique_trigrams': len(set(get_ngrams(a_tokens, 3))),
                'type_token_ratio': len(set(a_tokens)) / len(a_tokens),
                'avg_sentence_length': np.mean([len(d['answer'].split()) for d in self.data])
            }
        }
        
        return metrics
    
    
    
    def generate_report(self, output_path: Path):
        """Generate comprehensive analysis report"""
        report = {
            'basic_statistics': self.compute_basic_statistics(),
            'linguistic_diversity': self.compute_linguistic_diversity()
        }
        
        # Save JSON report with ensure_ascii=False to preserve symbols
        with open(output_path / 'dataset_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        # Generate markdown report
        self._generate_markdown_report(report, output_path / 'dataset_report.md')
        
        # Generate visualizations
        self._generate_visualizations(report, output_path / 'figures')
        
        
        para_dist = report['basic_statistics'].get('paragraph_distribution', {})
        if para_dist:
            para_dist_file = output_path / 'paragraph_distribution.json'
            with open(para_dist_file, 'w', encoding='utf-8') as f:
                # Convert Counter to regular dict for JSON serialization
                para_dist_dict = dict(para_dist)
                json.dump(para_dist_dict, f, indent=2, ensure_ascii=False)
            
            # Also save as readable text file
            para_dist_txt = output_path / 'paragraph_distribution.txt'
            with open(para_dist_txt, 'w', encoding='utf-8') as f:
                f.write("Paragraph Distribution\n")
                f.write("=" * 60 + "\n\n")
                for para, count in sorted(para_dist.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{para:20} : {count:6d} samples\n")
            
            self.logger.info(f"Saved paragraph distribution to:")
            self.logger.info(f"  JSON: {para_dist_file}")
            self.logger.info(f"  Text: {para_dist_txt}")
        
        return report
    
    def _generate_markdown_report(self, report: Dict, output_path: Path):
        """Generate human-readable markdown report"""
        md_content = f"""# Dataset Analysis Report

        ## Basic Statistics
        - Total Samples: {report['basic_statistics']['total_samples']:,}
        - Unique Paragraphs: {report['basic_statistics']['unique_paragraphs']}

        ### Label Distribution
        - Correct: {report['basic_statistics']['label_distribution']['yes']:,}
        - Incorrect: {report['basic_statistics']['label_distribution']['no']:,}

        ### Question Length
        - Mean: {report['basic_statistics']['question_length']['mean']:.1f} words
        - Std: {report['basic_statistics']['question_length']['std']:.1f}
        - Range: [{report['basic_statistics']['question_length']['min']}, {report['basic_statistics']['question_length']['max']}]

        ### Answer Length
        - Mean: {report['basic_statistics']['answer_length']['mean']:.1f} words
        - Std: {report['basic_statistics']['answer_length']['std']:.1f}
        - Range: [{report['basic_statistics']['answer_length']['min']}, {report['basic_statistics']['answer_length']['max']}]

        ## Linguistic Diversity
        ### Questions
        - Unique Unigrams: {report['linguistic_diversity']['questions']['unique_unigrams']:,}
        - Unique Bigrams: {report['linguistic_diversity']['questions']['unique_bigrams']:,}
        - Type-Token Ratio: {report['linguistic_diversity']['questions']['type_token_ratio']:.3f}

        ### Answers
        - Unique Unigrams: {report['linguistic_diversity']['answers']['unique_unigrams']:,}
        - Unique Bigrams: {report['linguistic_diversity']['answers']['unique_bigrams']:,}
        - Type-Token Ratio: {report['linguistic_diversity']['answers']['type_token_ratio']:.3f}

        """
        
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    def _generate_visualizations(self, report: Dict, output_dir: Path):
        """Generate visualization plots for the paper"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Label Distribution Pie Chart
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        labels = list(report['basic_statistics']['label_distribution'].keys())
        sizes = list(report['basic_statistics']['label_distribution'].values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Label Distribution')
        plt.savefig(output_dir / 'label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Text Length Distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Question lengths
        q_lengths = [len(d['question'].split()) for d in self.data]
        ax1.hist(q_lengths, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Question Length (words)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Question Length Distribution')
        ax1.axvline(np.mean(q_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(q_lengths):.1f}')
        ax1.legend()
        
        # Answer lengths
        a_lengths = [len(d['answer'].split()) for d in self.data]
        ax2.hist(a_lengths, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Answer Length (words)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Answer Length Distribution')
        ax2.axvline(np.mean(a_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(a_lengths):.1f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'text_length_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {output_dir}")


# ============= Model Evaluation =============

class ModelEvaluator:
    """Evaluate baseline models on the dataset"""
    
    def __init__(self, dataset_path: Path, model_name: str = "bert-base-uncased"):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.data = self.load_dataset()
        self.rouge = Rouge()
        
    
    def load_dataset(self) -> List[Dict]:
        """
        Load from QA dataset, safely handling JSON decoding errors for each line.
        """
        data = []
        
        # Check if the path is valid before trying to open
        if not os.path.exists(self.dataset_path):
            logging.error(f"Dataset file not found at: {self.dataset_path}")
            return data

        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                # We use enumerate to get the line number for better error reporting
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue # Skip empty lines

                    try:
                        # CRITICAL FIX: The try/except block must be around the 
                        # specific operation that can fail (json.loads)
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Log the error with the line number and a snippet of the problematic data
                        logging.error(
                            f"JSON Decode Error on line {i + 1} of dataset file: {e}. "
                            f"Skipping malformed line starting with: '{line[:80]}...'"
                        )
                        # The loop continues to the next line after logging the error
                    except Exception as e:
                        logging.error(f"An unexpected error occurred processing line {i + 1}: {e}")

        except Exception as e:
            # Handles errors opening or reading the file itself
            logging.critical(f"A file reading error occurred during dataset loading: {e}")
            return data

        logging.info(f"Successfully loaded {len(data)} valid samples.")
        return data

    
    def evaluate_generation_quality(self) -> Dict[str, float]:
        """Evaluate the quality of generated Q&A pairs"""
        metrics = {
            'valid_json_ratio': 0,
            'citation_presence_ratio': 0,
            'avg_bleu_score': 0,
            'avg_rouge_l': 0,
            'question_diversity': 0,
            'answer_diversity': 0
        }
        
        valid_count = 0
        citation_count = 0
        bleu_scores = []
        rouge_scores = []
        cosine_scores = []
        
        all_questions = []
        all_answers = []
        
        for item in self.data:
            # Check valid structure
            if all(key in item for key in ['question', 'answer', 'label', 'paragraph_id','paragraph','sentence']):
                valid_count += 1
            
            # Check citation presence
            if item.get('paragraph_id'):
                citation_count += 1
            
            all_questions.append(item['question'])
            all_answers.append(item['answer'])
            
            # Calculate text similarity between Q&A and source
            
            # BLEU score
            context1 = item['question'] + " " + item['answer']
            context2 = item['paragraph'] + " " + item['sentence']
            reference = context2.split()
            hypothesis = context1.split()
            bleu = sentence_bleu([reference], hypothesis)
            bleu_scores.append(bleu)
            
            # ROUGE score
            rouge_score = self.rouge.get_scores(context1, context2)[0]
            rouge_scores.append(rouge_score['rouge-l']['f'])

            
            
        
        
        # Calculate metrics
        metrics['valid_json_ratio'] = valid_count / len(self.data)
        metrics['citation_presence_ratio'] = citation_count / len(self.data)
        
        if bleu_scores:
            metrics['avg_bleu_score'] = np.mean(bleu_scores)
        
        if rouge_scores:
            metrics['avg_rouge_l'] = np.mean(rouge_scores)
        
        # Calculate diversity (unique n-grams / total n-grams)
        def calculate_diversity(texts, n=3):
            all_ngrams = []
            for text in texts:
                words = text.lower().split()
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
                all_ngrams.extend(ngrams)
            return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0
        
        metrics['question_diversity'] = calculate_diversity(all_questions)
        metrics['answer_diversity'] = calculate_diversity(all_answers)
        
        return metrics
    
    def cross_validate_baseline(self, n_folds: int = 5) -> Dict[str, float]:
        """Run cross-validation on baseline model"""
        
        
        # Prepare data
        X = [f"{item['question']} {item['answer']}" for item in self.data]
        y = [1 if item['label'] == 'yes' else 0 for item in self.data]
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]
            
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_val)
            
            accuracy = accuracy_score(y_val, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, predictions, average='binary'
            )
            
            scores.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        # Average scores
        avg_scores = {
            'baseline_model': 'LogisticRegression + TF-IDF',
            'accuracy': np.mean([s['accuracy'] for s in scores]),
            'precision': np.mean([s['precision'] for s in scores]),
            'recall': np.mean([s['recall'] for s in scores]),
            'f1': np.mean([s['f1'] for s in scores]),
            'std_f1': np.std([s['f1'] for s in scores])
        }
        
        return avg_scores
    
    def generate_evaluation_report(self, output_path: Path):
        """Generate comprehensive evaluation report"""
        logging.info("Starting comprehensive evaluation...")
        
        report = {
            'generation_quality': self.evaluate_generation_quality(),
            'baseline_performance': self.cross_validate_baseline()
    
        }
        
        # Save JSON report
        with open(output_path / 'evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        
        
        # Generate markdown report
        self._generate_evaluation_markdown(report, output_path / 'evaluation_report.md')
        
        logging.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def _generate_evaluation_markdown(self, report: Dict, output_path: Path):
        """Generate markdown evaluation report"""
        md_content = f"""# Model Evaluation Report

        ## Generation Quality
        - Valid JSON Ratio: {report['generation_quality']['valid_json_ratio']:.1%}
        - Citation Presence: {report['generation_quality']['citation_presence_ratio']:.1%}
        - Avg BLEU Score: {report['generation_quality']['avg_bleu_score']:.3f}
        - Avg ROUGE-L: {report['generation_quality']['avg_rouge_l']:.3f}
        - Question Diversity: {report['generation_quality']['question_diversity']:.3f}
        - Answer Diversity: {report['generation_quality']['answer_diversity']:.3f}

        ## Baseline Performance (5-fold CV)
        - Accuracy: {report['baseline_performance']['accuracy']:.3f}
        - Precision: {report['baseline_performance']['precision']:.3f}
        - Recall: {report['baseline_performance']['recall']:.3f}
        - F1 Score: {report['baseline_performance']['f1']:.3f} ± {report['baseline_performance']['std_f1']:.3f}
        """
                
        with open(output_path, 'w') as f:
            f.write(md_content)


# ============= Human Evaluation Sampling =============

class HumanEvaluationSampler:
    """Sample and prepare data for human evaluation"""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.data = self.load_dataset()
    
    def load_dataset(self) -> List[Dict]:
        """
        Load from QA dataset, safely handling JSON decoding errors for each line.
        """
        data = []
        
        # Check if the path is valid before trying to open
        if not os.path.exists(self.dataset_path):
            logging.error(f"Dataset file not found at: {self.dataset_path}")
            return data

        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                # We use enumerate to get the line number for better error reporting
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue # Skip empty lines

                    try:
                        # CRITICAL FIX: The try/except block must be around the 
                        # specific operation that can fail (json.loads)
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Log the error with the line number and a snippet of the problematic data
                        logging.error(
                            f"JSON Decode Error on line {i + 1} of dataset file: {e}. "
                            f"Skipping malformed line starting with: '{line[:80]}...'"
                        )
                        # The loop continues to the next line after logging the error
                    except Exception as e:
                        logging.error(f"An unexpected error occurred processing line {i + 1}: {e}")

        except Exception as e:
            # Handles errors opening or reading the file itself
            logging.critical(f"A file reading error occurred during dataset loading: {e}")
            return data

        logging.info(f"Successfully loaded {len(data)} valid samples.")
        return data
        

    
    def sample_for_evaluation(
        self, 
        n_samples: int = 200,
        stratified: bool = True
    ) -> List[Dict]:
        """Sample examples for human evaluation"""
        
        if stratified:
            # Stratified sampling by question type and label
            samples = []
            
            # Group by question type and label
            grouped = defaultdict(list)
            for item in self.data:
                key = (item.get('question_type', 'unknown'), item['label'])
                grouped[key].append(item)
            
            # Sample proportionally from each group
            samples_per_group = max(1, n_samples // len(grouped))
            
            for key, items in grouped.items():
                n_from_group = min(len(items), samples_per_group)
                samples.extend(random.sample(items, n_from_group))
            
            # If we need more samples, add randomly
            if len(samples) < n_samples:
                remaining = n_samples - len(samples)
                unused = [item for item in self.data if item not in samples]
                samples.extend(random.sample(unused, min(remaining, len(unused))))
            
            return samples[:n_samples]
        else:
            # Random sampling
            return random.sample(self.data, min(n_samples, len(self.data)))
    
    def prepare_annotation_file(
        self,
        output_path: Path,
        n_samples: int = 200
    ):
        """Prepare CSV file for human annotation"""
        samples = self.sample_for_evaluation(n_samples)
        
        # Prepare annotation format
        annotation_data = []
        for idx, sample in enumerate(samples, 1):
            annotation_data.append({
                'id': idx,
                'question': sample['question'],
                'answer': sample['answer'],
                'label': sample['label'],
                'source_paragraph': sample['paragraph'][:500] + '...',
                'question_type': sample.get('question_type', 'unknown'),
                'factual_correctness': '',  # To be filled by annotator
                'citation_accuracy': '',    # To be filled by annotator
                'question_quality': '',     # To be filled by annotator
                'legal_soundness': '',      # To be filled by annotator
                'notes': ''                 # Additional comments
            })
        
        # Save as CSV
        df = pd.DataFrame(annotation_data)
        csv_path = output_path / 'human_evaluation_samples.csv'
        df.to_csv(csv_path, index=False)
        
        # Also save as JSON for reference
        json_path = output_path / 'human_evaluation_samples.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Prepared {len(samples)} samples for human evaluation")
        logging.info(f"CSV saved to: {csv_path}")
        logging.info(f"JSON saved to: {json_path}")
        
        return samples


# ============= Main Execution =============

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation suite for Citation-based Legal Q&A Dataset"
    )
    
    parser.add_argument('--base_model', type=str, default="bert-base-uncased",
                       help="Base model name for evaluation")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device for model evaluation (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--dataset_path', type=str, default='qa_combined_all_final.jsonl',
                       help="Path to JSONL dataset")
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help="Directory for evaluation outputs")
    parser.add_argument('--training_args_out', type=str, default='/fast_storage/siddig/driving-license/LLM_evaluator/',
                       help="Directory for training arguments")
    parser.add_argument('--fine_tuned_dir', type=str, default='./fine_tuned_models',
                       help="Directory for fine-tuned models")
    parser.add_argument('--run_analysis', action='store_true',
                       help="Run dataset analysis")
    parser.add_argument('--run_evaluation', action='store_true',
                       help="Run model evaluation")
    parser.add_argument('--run_fine_tuning', action='store_true',
                       help="Run fine-tuning of baseline models")
    parser.add_argument('--prepare_human_eval', action='store_true',
                       help="Prepare human evaluation samples")
    parser.add_argument('--n_human_samples', type=int, default=200,
                       help="Number of samples for human evaluation")
    parser.add_argument('--cache_dir', type=str, default='./fast_storage/siddig/driving-license/HF_models/',
                       help="Cache directory for Hugging Face models")
    parser.add_argument('--access_token', type=str, default="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",
                       help="Hugging Face access token for private models")
    parser.add_argument('--add_paragraph', action='store_true',
                       help="Whether to add paragraph context during fine-tuning")
    parser.add_argument('--paragraph_id_clustering', action='store_true',
                       help="Whether to use paragraph ID clustering during fine-tuning")
    parser.add_argument('--binary_classification', action='store_true',
                       help="Whether to perform binary classification during fine-tuning")
    parser.add_argument('--zero_shot_eval', action='store_true',
                       help="Whether to perform zero-shot evaluation")
    parser.add_argument('--train_size', type=float, default=0.4,
                       help="Training set size proportion")
    parser.add_argument('--val_size', type=float, default=0.1,
                       help="Validation set size proportion")
    parser.add_argument('--test_size', type=float, default=0.5,
                       help="Test set size proportion")
    parser.add_argument('--epochs', type=int, default=5,
                       help="Number of epochs for fine-tuning")

    print("#################### Parsed Arguments ####################")
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("##########################################################")
    
    # Correct logic: If CUDA is NOT available, raise the error.
    if not torch.cuda.is_available():
        raise ValueError("GPU is NOT available. Please check your CUDA installation.")
    else:
        print(f"✔ GPU detected: {torch.cuda.get_device_name(0)}")
    
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fine_tuned_dir = Path(args.fine_tuned_dir)
    fine_tuned_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = Path(args.dataset_path)
    # list GPU devices for verification
    if torch.cuda.is_available():
        logging.info(f"Available GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("No GPU devices available, using CPU.")

    # Run analyses based on flags
    if args.run_analysis:
        logging.info("Running dataset analysis...")
        analyzer = DatasetAnalyzer(dataset_path)
        report = analyzer.generate_report(output_dir)
        logging.info("Dataset analysis complete")
    
    if args.run_evaluation:
        logging.info("Running model evaluation...")
        evaluator = ModelEvaluator(dataset_path)
        eval_report = evaluator.generate_evaluation_report(output_dir)
        logging.info("Model evaluation complete")
        logging.info(f"Evaluation results: {eval_report}")

    if args.run_fine_tuning:
        logging.info("Running fine-tuning...")
        QA_trainer = BaselineLLMEvaluator(dataset_path, model_name=args.base_model,fine_tuned_dir=fine_tuned_dir,
                                          access_token=args.access_token, 
                                          add_paragraph=args.add_paragraph,
                                         paragraph_id_clustering=args.paragraph_id_clustering,
                                            binary_classification=args.binary_classification,
                                          cache_dir=args.cache_dir, 
                                          device=args.device,
                                          training_args_out=args.training_args_out)

        QA_trainer.run_evaluation(output_dir,zero_shot_eval=args.zero_shot_eval,
                                  train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, epochs=args.epochs)
        
        logging.info("Fine-tuning and evaluation complete")
        

    if args.prepare_human_eval:
        logging.info("Preparing human evaluation samples...")
        sampler = HumanEvaluationSampler(dataset_path)
        sampler.prepare_annotation_file(output_dir, args.n_human_samples)
        logging.info("Human evaluation preparation complete")
    
    logging.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()