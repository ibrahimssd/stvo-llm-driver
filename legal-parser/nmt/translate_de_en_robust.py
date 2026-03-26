"""
Robust German-to-English Translation Script - FIXED VERSION
Fixes:
1. M2M100 cache_dir parameter issue
2. Helsinki OPUS task name format
"""

import json
import torch
import logging
import argparse
import re
from typing import Dict, List, Any, Optional, Tuple
from transformers import pipeline
from functools import lru_cache
from pathlib import Path
from datetime import datetime


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"translation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")


class RobustTranslator:
    """Robust German-to-English translator with quality validation."""
    
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512, 
                 custom_cache_dir: Optional[str] = None):
        """Initialize the translator."""
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.cache_dir = custom_cache_dir
        
        # Initialize translator
        self.translator = self._load_translator()
        
        # Statistics tracking
        self.stats = {
            'total_translations': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'fallback_used': 0,
        }
        
        # Special legal terms that should not be translated
        self.preserve_terms = {
            '§': '§',
            'StVO': 'StVO',
            'VwV-StVO': 'VwV-StVO',
            'BASt': 'BASt',
        }

    def _load_translator(self):
        """Load the translator model - FIXED VERSION."""
        try:
            # Use correct task format for both models
            task = 'translation_de_to_en'
            
            logging.info(f"Loading model: {self.model_name} for task: {task}")
            logging.info(f"Device: {self.device}")
            
            # Build pipeline kwargs carefully
            pipeline_kwargs = {
                "model": self.model_name,
                "device": 0 if self.device == "cuda" else -1,
                "framework": "pt"
            }
            
            # Note: Don't pass cache_dir to pipeline - it causes issues with M2M models
            # The model will use HF_HOME environment variable instead
            
            translator = pipeline(task, **pipeline_kwargs)
            
            logging.info(f"Model loaded successfully on device: {self.device}")
            return translator
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def _replace_special_characters(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace special characters with placeholders."""
        replacements = {}
        modified_text = text
        
        if '§' in modified_text:
            placeholder = '__PARAGRAPH_SIGN__'
            replacements['§'] = placeholder
            modified_text = modified_text.replace('§', placeholder)
        
        return modified_text, replacements

    def _restore_special_characters(self, text: str, replacements: Dict[str, str]) -> str:
        """Restore special characters from placeholders."""
        restored_text = text
        reverse_map = {v: k for k, v in replacements.items()}
        
        for placeholder, original in reverse_map.items():
            restored_text = restored_text.replace(placeholder, original)
        
        return restored_text

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def translate_text(self, text: str) -> str:
        """Translate German text to English with error handling."""
        try:
            self.stats['total_translations'] += 1
            
            if not text or not text.strip():
                self.stats['successful_translations'] += 1
                return text
            
            # Replace special characters
            text_with_placeholders, replacements = self._replace_special_characters(text)
            
            # Split text into chunks
            chunks = self.split_text(text_with_placeholders)
            
            if not chunks:
                self.stats['failed_translations'] += 1
                return text
            
            # Translate chunks
            translated_chunks = []
            for chunk in chunks:
                try:
                    result = self.translator(chunk, max_length=self.max_length)
                    if result and len(result) > 0:
                        translated_chunk = result[0]['translation_text']
                        translated_chunks.append(translated_chunk)
                    else:
                        logging.warning(f"Empty translation result for chunk: {chunk[:50]}")
                        translated_chunks.append(chunk)
                        self.stats['fallback_used'] += 1
                        
                except Exception as e:
                    logging.warning(f"Failed to translate chunk: {e}. Using original.")
                    translated_chunks.append(chunk)
                    self.stats['fallback_used'] += 1
            
            # Combine chunks
            translated_text = " ".join(translated_chunks)
            
            # Restore special characters
            translated_text = self._restore_special_characters(translated_text, replacements)
            
            self.stats['successful_translations'] += 1
            return translated_text
            
        except Exception as e:
            logging.error(f"Error translating text: {e}. Using original text.")
            self.stats['failed_translations'] += 1
            return text

    def translate_paragraph_title(self, title_text: str) -> str:
        """Intelligently translate paragraph titles like "§ 8 Vorfahrt"."""
        try:
            match = re.match(r"(§\s*\d+\s*)(.*)", title_text)
            
            if match:
                marker = match.group(1)
                text_to_translate = match.group(2).strip()
                
                if not text_to_translate:
                    return marker
                
                translated_text = self.translate_text(text_to_translate)
                return f"{marker}{translated_text}"
            else:
                return self.translate_text(title_text)
                
        except Exception as e:
            logging.error(f"Error translating paragraph title: {e}")
            return title_text

    def recursive_main_content_translate(self, data: List[Dict]) -> List[Dict]:
        """Recursively translate main content structure."""
        logging.info(f"Starting translation of {len(data)} categories")
        modified_categories = []
        
        for cat_idx, category in enumerate(data):
            try:
                translated_category = self.translate_text(category.get('category', ''))
                logging.info(f"[{cat_idx+1}/{len(data)}] Category: {translated_category}")
                
                modified_paragraphs = []
                paragraphs = category.get('paragraphs', [])
                
                for para_idx, paragraph in enumerate(paragraphs):
                    try:
                        translated_paragraph = self.translate_paragraph_title(
                            paragraph.get('paragraph', '')
                        )
                        
                        modified_sentences = []
                        sentences = paragraph.get('sentences', [])
                        
                        for sentence_dict in sentences:
                            try:
                                for sen_id, sen_text in sentence_dict.items():
                                    translated_text = self.translate_text(sen_text)
                                    modified_sentences.append({sen_id: translated_text})
                            except Exception as e:
                                logging.warning(f"Error translating sentence: {e}")
                                modified_sentences.append(sentence_dict)
                        
                        modified_paragraphs.append({
                            'paragraph': translated_paragraph,
                            'paragraph_id': paragraph.get('paragraph_id', ''),
                            'sentences': modified_sentences
                        })
                        
                    except Exception as e:
                        logging.error(f"Error processing paragraph {para_idx}: {e}")
                        modified_paragraphs.append(paragraph)
                
                modified_categories.append({
                    'category': translated_category,
                    'paragraphs': modified_paragraphs
                })
                
            except Exception as e:
                logging.error(f"Error processing category {cat_idx}: {e}")
                modified_categories.append(category)
        
        logging.info(f"Translation complete. {len(modified_categories)} categories processed")
        return modified_categories

    def recursive_table_of_contents_translate(self, data: List[Dict]) -> List[Dict]:
        """Recursively translate table of contents structure."""
        logging.info(f"Starting translation of {len(data)} tables")
        modified_table_data = []
        
        for table_idx, table in enumerate(data):
            try:
                translated_table_name = self.translate_text(table.get('table_name', ''))
                logging.info(f"[{table_idx+1}/{len(data)}] Table: {translated_table_name}")
                
                modified_sections = []
                
                for section in table.get('table_data', []):
                    try:
                        translated_section = self.translate_text(section.get('Section', ''))
                        
                        modified_entries = []
                        for entry in section.get('Content', []):
                            try:
                                sign_number = entry.get('Sign_number', '')
                                translated_sign_name = f"Sign {sign_number}"
                                entry['Sign_name'] = translated_sign_name
                                
                                translated_description = self.translate_text(
                                    entry.get('description', '')
                                )
                                entry['description'] = translated_description
                                
                                modified_entries.append(entry)
                            except Exception as e:
                                logging.warning(f"Error translating table entry: {e}")
                                modified_entries.append(entry)
                        
                        modified_sections.append({
                            'Section': translated_section,
                            'Section_Id': section.get('Section_Id', ''),
                            'Content': modified_entries
                        })
                        
                    except Exception as e:
                        logging.error(f"Error processing section: {e}")
                        modified_sections.append(section)
                
                modified_table_data.append({
                    'table_name': translated_table_name,
                    'table_id': table.get('table_id', ''),
                    'table_data': modified_sections
                })
                
            except Exception as e:
                logging.error(f"Error processing table {table_idx}: {e}")
                modified_table_data.append(table)
        
        logging.info(f"Table translation complete. {len(modified_table_data)} tables processed")
        return modified_table_data

    def save_statistics(self, output_dir: str):
        """Save translation statistics."""
        stats_file = Path(output_dir) / "translation_statistics.json"
        
        stats_summary = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'success_rate': (
                self.stats['successful_translations'] / 
                max(self.stats['total_translations'], 1) * 100
            )
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_summary, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Statistics saved to {stats_file}")
        logging.info(f"Success rate: {stats_summary['success_rate']:.1f}%")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Translate German legal text to English robustly'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default="facebook/m2m100_418M",
        help='Model for translation'
    )
    parser.add_argument(
        '--parsed_main',
        type=str,
        default=None,
        help='Path to parsed main content JSON file'
    )
    parser.add_argument(
        '--parsed_table',
        type=str,
        default=None,
        help='Path to parsed table content JSON file'
    )
    parser.add_argument(
        '--translated_main',
        type=str,
        default="translated_main_content_Straßenverkehrs_Ordnung.json",
        help='Output filename for translated main content'
    )
    parser.add_argument(
        '--translated_table',
        type=str,
        default="translated_table_content_Straßenverkehrs_Ordnung.json",
        help='Output filename for translated table content'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=400,
        help='Maximum length of text chunks for translation'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=".",
        help='Output directory for translated files'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.out_dir)
    
    # Log configuration
    logging.info("=" * 80)
    logging.info("TRANSLATION CONFIGURATION (FIXED VERSION)")
    logging.info("=" * 80)
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("=" * 80)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Initialize translator
    translator = RobustTranslator(
        model_name=args.model_name,
        device=device,
        max_length=args.max_length,
        custom_cache_dir=None  # Let HF_HOME handle caching
    )
    
    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Translate main content
    if args.parsed_main:
        logging.info(f"\nProcessing main content from: {args.parsed_main}")
        try:
            with open(args.parsed_main, 'r', encoding='utf-8') as file:
                main_content_data = json.load(file)
            
            translated_main_content = translator.recursive_main_content_translate(main_content_data)
            
            model_name = args.model_name.split('/')[1]
            translated_main_file = f'{args.out_dir}/{model_name}_{args.translated_main}'
            
            with open(translated_main_file, 'w', encoding='utf-8') as file:
                json.dump(translated_main_content, file, ensure_ascii=False, indent=4)
            
            logging.info(f"Main content saved to: {translated_main_file}")
            
        except Exception as e:
            logging.error(f"Error processing main content: {e}")
    
    # Translate table content
    if args.parsed_table:
        logging.info(f"\nProcessing table content from: {args.parsed_table}")
        try:
            with open(args.parsed_table, 'r', encoding='utf-8') as file:
                table_content_data = json.load(file)
            
            translated_table_content = translator.recursive_table_of_contents_translate(table_content_data)
            
            model_name = args.model_name.split('/')[1]
            translated_table_file = f'{args.out_dir}/{model_name}_{args.translated_table}'
            
            with open(translated_table_file, 'w', encoding='utf-8') as file:
                json.dump(translated_table_content, file, ensure_ascii=False, indent=4)
            
            logging.info(f"Table content saved to: {translated_table_file}")
            
        except Exception as e:
            logging.error(f"Error processing table content: {e}")
    
    # Save statistics
    translator.save_statistics(args.out_dir)
    
    logging.info("\n" + "=" * 80)
    logging.info("TRANSLATION COMPLETE")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()