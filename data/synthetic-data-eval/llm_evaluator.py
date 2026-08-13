from torch.utils.data import Dataset
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from pathlib import Path
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import logging
from sklearn.model_selection import train_test_split
import random
import os
from collections import Counter
from report import EvaluationReportGenerator 

# ============= Models Baseline Evaluation =============
def set_seed(seed_value=42):
    """Set seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    logging.info(f"Global seed set to {seed_value}")


class QADataset(Dataset):
    """PyTorch Dataset for Q&A verification.
    
    Supports two classification modes:
    1. Label-based: Binary classification (yes/no labels)
    2. Paragraph ID clustering: Multi-class classification based on paragraph_id (§ section numbers)
    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, 
                 add_paragraph: bool = False, paragraph_id_clustering: bool = False, binary_classification: bool = False,
                 paragraph_id_to_label: Optional[Dict[str, int]] = None, num_classes: int = 2):
        """
        Args:
            data: List of dataset records
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            add_paragraph: Include paragraph in input text
            paragraph_id_clustering: If True, classify by paragraph_id; if False, use yes/no labels
            paragraph_id_to_label: Mapping from paragraph_id to numeric class label
            num_classes: Number of output classes
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_paragraph = add_paragraph
        self.paragraph_id_clustering = paragraph_id_clustering
        self.binary_classification = binary_classification
        self.paragraph_id_to_label = paragraph_id_to_label or {}
        self.num_classes = num_classes
        
        logging.info(f"QADataset initialized - Clustering mode: {paragraph_id_clustering}, "
                    f"Num classes: {num_classes}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        # sentence = item['sentence']
        context = item['paragraph']
        
        if self.add_paragraph:
            text = f"{context} [SEP] Question: {question} [SEP] Answer: {answer}"
        else:
            text = f"Question: {question} [SEP] Answer: {answer}"
        
        # 🌟 FEATURE: Support for paragraph_id clustering vs label-based classification
        if self.paragraph_id_clustering:
            # Classify based on paragraph_id (e.g., "§ 45" → class label)
            paragraph_id = item.get('paragraph_id', 'unknown')
            label = self.paragraph_id_to_label.get(paragraph_id, 0)
            logging.debug(f"Clustering mode: paragraph_id={paragraph_id} → label={label}")
        elif self.binary_classification:
            # Binary classification based on correct/incorrect labels
            # Original: binary classification based on correct/incorrect labels
            label = 1 if item.get('label') == 'correct' else 0
        else:
            raise ValueError("Either paragraph_id_clustering or binary_classification must be True.")
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BaselineLLMEvaluator:
    """Evaluate baseline classification models - supports both label and paragraph_id clustering modes"""
    
    def __init__(self, dataset_path: Path, model_name: str = "bert-base-uncased",fine_tuned_dir: Path = None, seed: int = 42, 
                 add_paragraph: bool = False, paragraph_id_clustering: bool = False, binary_classification: bool = False,
                 access_token: str = None, cache_dir: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 training_args_out: str = None):
        """
        Initialize evaluator with support for different classification modes.
        
        Args:
            paragraph_id_clustering: If True, classify by paragraph_id clusters; 
                                        if False, use binary correct/incorrect labels
        """
        self.seed = seed
        set_seed(seed)
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.fine_tuned_model_dir = fine_tuned_dir
        self.add_paragraph = add_paragraph
        self.paragraph_id_clustering = paragraph_id_clustering
        self.binary_classification = binary_classification
        self.access_token = access_token
        self.cache_dir = cache_dir
        self.training_args_out = training_args_out
        self.device = device
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.data = self.load_dataset()
        
        # 🌟 NEW: Build paragraph_id to label mapping if using clustering mode
        self.paragraph_id_to_label = {}
        if self.paragraph_id_clustering:
            self.paragraph_id_to_label, self.num_classes = self._build_paragraph_id_mapping()
            logging.info(f"Paragraph ID clustering mode: {self.num_classes} classes detected")
        elif self.binary_classification:
            self.num_classes = 2
            logging.info("Label-based classification mode: Binary classification (correct/incorrect)")
        else:
            raise ValueError("Either paragraph_id_clustering or binary_classification must be True.")
        

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                token=self.access_token, 
                cache_dir=self.cache_dir
            )
             
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                token=self.access_token,
                cache_dir=self.cache_dir,
                num_labels=self.num_classes,
                trust_remote_code=True
            )

            self.model.to(self.device)
            logging.info(f"Model loaded with {self.num_classes} output classes")
        
        except Exception as e:
            logging.error(f"Error loading model from {self.model_name}: {e}")
            logging.info("Attempting to load model from fine-tuned directory...")
            try:
                model_checkpoint = Path(self.fine_tuned_model_dir) / self.model_name
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_checkpoint,
                    token=self.access_token, 
                    cache_dir=self.cache_dir
                    )
                
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_checkpoint,
                    token=self.access_token,
                    cache_dir=self.cache_dir,
                    num_labels=self.num_classes,
                    trust_remote_code=True
                ).to(self.device)
                logging.info(f"Model loaded from fine-tuned directory with {self.num_classes} output classes")
            
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                raise

    def load_dataset(self) -> List[Dict]:
        """Load dataset with robust error handling"""
        data = []
        error_count = 0
        recovered_count = 0
        
        if not os.path.exists(self.dataset_path):
            logging.error(f"Dataset file not found at: {self.dataset_path}")
            return data

        logging.info(f"Loading dataset from: {self.dataset_path}")
        
        try:
            with open(self.dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        data.append(record)
                    except json.JSONDecodeError as e:
                        error_count += 1
                        
                        # Try recovery: find last valid closing brace
                        recovered = False
                        for end_pos in range(len(line) - 1, max(0, len(line) - 1000), -1):
                            if line[end_pos] == '}':
                                try:
                                    attempt = line[:end_pos + 1]
                                    record = json.loads(attempt)
                                    data.append(record)
                                    recovered_count += 1
                                    recovered = True
                                    break
                                except json.JSONDecodeError:
                                    continue
                        
                        if not recovered and error_count <= 5:
                            logging.debug(f"Line {line_num}: Could not recover from JSON error")
        except Exception as e:
            logging.critical(f"Error reading file: {e}")
            return data

        logging.info(f"Successfully loaded {len(data)} valid samples")
        if error_count > 0:
            logging.info(f"Skipped {error_count} malformed lines ({recovered_count} recovered)")
        
        return data

    def _build_paragraph_id_mapping(self) -> tuple:
        """Build mapping from paragraph_id to numeric class label.
        
        Returns:
            tuple: (paragraph_id_to_label dict, num_classes int)
        """
        paragraph_ids = [record.get('paragraph_id', 'unknown') for record in self.data]
        unique_ids = sorted(set(paragraph_ids))
        
        mapping = {pid: idx for idx, pid in enumerate(unique_ids)}
        num_classes = len(unique_ids)
        
        # Log statistics
        id_counts = Counter(paragraph_ids)
        logging.info(f"Found {num_classes} unique paragraph IDs:")
        for pid, count in id_counts.most_common(10):
            logging.info(f"  {pid}: {count} samples")
        
        return mapping, num_classes

    def split_data(self, train_size=0.4, val_size=0.1, test_size=0.5, seed=42) -> tuple:
        """Split data into train, validation, and test sets."""
        assert train_size + val_size + test_size == 1.0, "Sizes must sum to 1.0"
        
        train_data, temp_data = train_test_split(
            self.data, 
            train_size=train_size, 
            random_state=seed
        )
        
        relative_val_size = val_size / (val_size + test_size)
        val_data, test_data = train_test_split(
            temp_data, 
            train_size=relative_val_size, 
            random_state=seed
        )
        
        logging.info(f"Data Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Log label/clustering distribution
        if self.paragraph_id_clustering:
            self._log_clustering_distribution(train_data, val_data, test_data)
        elif self.binary_classification:
            self._log_label_distribution(train_data, val_data, test_data)
        else:
            raise ValueError("Either paragraph_id_clustering or binary_classification must be True.")
        
        return train_data, val_data, test_data

    def _log_label_distribution(self, train_data, val_data, test_data):
        """Log label distribution (correct/incorrect) across splits"""
        for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            correct_count = sum(1 for r in split_data if r.get('label') == 'correct')
            incorrect_count = len(split_data) - correct_count
            logging.info(f"  {split_name}: Correct={correct_count}, Incorrect={incorrect_count}")

    def _log_clustering_distribution(self, train_data, val_data, test_data):
        """Log paragraph_id distribution across splits"""
        for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            ids = [r.get('paragraph_id', 'unknown') for r in split_data]
            id_counts = Counter(ids)
            logging.info(f"  {split_name}: {len(id_counts)} classes, distribution={dict(id_counts)}")

    def evaluate_zero_shot(self, data: List[Dict]) -> Dict[str, float]:
        """Zero-shot evaluation"""
        logging.info("Starting zero-shot evaluation...")
        
        # Use a minimal TrainingArguments for evaluation only
        eval_args = TrainingArguments(
            output_dir=f'{self.training_args_out}/zero_shot_results',
            per_device_eval_batch_size=16,
            do_train=False,
            do_eval=True,
            report_to='none',
            remove_unused_columns=False,
        )

        zero_shot_trainer = Trainer(
            model=self.model,
            args=eval_args,
            compute_metrics=self._compute_metrics
        )
        
        logging.info(f"Evaluating on {len(data)} samples...")
        results = self.evaluate_model(zero_shot_trainer, data)
        logging.info("Zero-Shot Evaluation Complete")
        
        return results
    
    def _compute_metrics(self, p) -> Dict[str, float]:
        """Compute evaluation metrics - handles both binary and multi-class"""
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        
        # Determine average method based on number of classes
        avg_method = 'binary' if self.num_classes == 2 else 'macro'
        
        accuracy = accuracy_score(labels, preds)
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average=avg_method, zero_division=0)
        recall = recall_score(labels, preds, average=avg_method, zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
        
        return {
            'eval_accuracy': accuracy,
            'eval_balanced_accuracy': balanced_accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1_macro': f1_macro,
            'eval_f1_micro': f1_micro,
            'num_classes': self.num_classes
        }
    
    def fine_tune_model(self, train_data: List[Dict], val_data: List[Dict], epochs: int = 3) -> Trainer:
        """Fine-tune the model"""
        
        train_dataset = QADataset(
            train_data, 
            self.tokenizer, 
            add_paragraph=self.add_paragraph,
            paragraph_id_clustering=self.paragraph_id_clustering,
            binary_classification=self.binary_classification,
            paragraph_id_to_label=self.paragraph_id_to_label,
            num_classes=self.num_classes
        )
        
        val_dataset = QADataset(
            val_data, 
            self.tokenizer, 
            add_paragraph=self.add_paragraph,
            paragraph_id_clustering=self.paragraph_id_clustering,
            binary_classification=self.binary_classification,
            paragraph_id_to_label=self.paragraph_id_to_label,
            num_classes=self.num_classes
        )
        

        # 🌟 IMPROVEMENT: Lower epochs for quicker feedback loop, unless proven necessary
        training_args = TrainingArguments(
            output_dir=f'{self.training_args_out}/fine_tuned_model',
            num_train_epochs=epochs,
            per_device_train_batch_size=16, # Increased batch size (if memory allows)
            per_device_eval_batch_size=16,
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_dir='./logs',
            logging_steps=100, # Increased logging steps
            load_best_model_at_end=True,
            # 🌟 IMPROVEMENT: Use F1-score as metric for potentially imbalanced datasets
            metric_for_best_model='eval_f1_macro',
            greater_is_better=True,
            report_to='none',
            seed=self.seed
        )
        
        trainer = Trainer(
            model=self.model.to(self.device),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        logging.info(f"Starting fine-tuning for {epochs} epochs...")
        trainer.train()
        
        return trainer
    
    def evaluate_model(self, trainer: Trainer, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate model on test data"""
        
        test_dataset = QADataset(
            test_data, 
            self.tokenizer,
            add_paragraph=self.add_paragraph,
            paragraph_id_clustering=self.paragraph_id_clustering,
            binary_classification=self.binary_classification,
            paragraph_id_to_label=self.paragraph_id_to_label,
            num_classes=self.num_classes
        )
        
        results = trainer.evaluate(eval_dataset=test_dataset)
        return results
    
    def run_evaluation(self, output_dir: Path, zero_shot_eval: bool = False,
                       epochs: int = 3, train_size=0.4, val_size=0.1, test_size=0.5) -> Dict:
        """Execute full evaluation pipeline"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        
        all_results = {}
        
        # Zero-shot evaluation
        if zero_shot_eval:
            logging.info("Running zero-shot evaluation...")
            zero_shot_results = self.evaluate_zero_shot(self.data)
            all_results['zero_shot_whole_dataset'] = zero_shot_results
            
        else:
            logging.info("Skipping zero-shot evaluation as per configuration.")
            # Split data
            train_data, val_data, test_data = self.split_data(train_size, val_size, test_size, self.seed)
            
            # Fine-tune
            logging.info("Starting fine-tuning...")
            trainer = self.fine_tune_model(train_data, val_data, epochs=epochs)
            
            # Evaluate
            logging.info("Evaluating on test set...")
            fine_tuned_results = self.evaluate_model(trainer, test_data)
            all_results['fine_tuned_test_set'] = fine_tuned_results
            
            # Save model
            model_save_path = f'{self.training_args_out}/fine_tuned_model/'
            mode_flag = "paragraph_id_clustering" if self.paragraph_id_clustering else "binary_classification"
            model_save_path = Path(model_save_path) / f'best_model_{self.model_name.replace("/", "_")}_{mode_flag}'
            model_save_path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(model_save_path))
            logging.info(f"Model saved to {model_save_path}")
            
        # Generate report
        # self.generate_report(output_dir, all_results) # old reporting 

        
        report_generator = EvaluationReportGenerator(
            model_name=self.model_name,
            num_classes=self.num_classes,
            paragraph_id_clustering=self.paragraph_id_clustering,
            add_paragraph=self.add_paragraph,
            paragraph_id_to_label=self.paragraph_id_to_label
        )
        report_generator.generate_report(output_dir, all_results)  
            
        return all_results
    
    def generate_report(self, output_path: Path, report: Dict):
        """Generate evaluation report"""
        logging.info("Generating evaluation report...")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 🌟 Include classification mode in report name
        mode_flag = "paragraph_id_clustering" if self.paragraph_id_clustering else "binary_classification"
        context_flag = "with_paragraph" if self.add_paragraph else "without_paragraph"
        # extract model name safe for filenames
        model_name = self.model_name.replace("/", "_")
        
        # Prepare output report
        output_report = {
            'mode': mode_flag,
            'context_flag': context_flag,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'paragraph_id_mapping': self.paragraph_id_to_label if self.paragraph_id_clustering else None
        }
        
        # Flatten nested dicts
        for key, value in report.items():
            if isinstance(value, dict):
                for metric_key, metric_value in value.items():
                    output_report[f"{key}_{metric_key}"] = metric_value
            else:
                output_report[key] = value
        
        # Save JSON
        report_path = output_path / f'evaluation_report_{mode_flag}_{model_name}_{context_flag}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(output_report, f, indent=2, default=str, ensure_ascii=False)
        logging.info(f"JSON report saved to {report_path}")
        
        # Save CSV
        csv_path = output_path / f'evaluation_report_{mode_flag}_{model_name}_{context_flag}.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Setting,Mode,Accuracy,Balanced Accuracy,Precision,Recall,F1 Macro,F1 Micro\n")
            
            for setting, metrics in report.items():
                if isinstance(metrics, dict):
                    values = [
                        mode_flag,
                        str(metrics.get('eval_accuracy', 'N/A')),
                        str(metrics.get('eval_balanced_accuracy', 'N/A')),
                        str(metrics.get('eval_precision', 'N/A')),
                        str(metrics.get('eval_recall', 'N/A')),
                        str(metrics.get('eval_f1_macro', 'N/A')),
                        str(metrics.get('eval_f1_micro', 'N/A'))
                    ]
                    f.write(f"{setting},{','.join(values)}\n")
        
        logging.info(f"CSV report saved to {csv_path}")

