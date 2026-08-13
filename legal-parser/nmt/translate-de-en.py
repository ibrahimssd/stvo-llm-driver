import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM # AutoModelForCausalLM for decoder-only LLMs
import torch
from functools import lru_cache
import logging
import argparse
import os # Import os for environment variables and path manipulation
import re # Import re for symbol handling and improved Sign_name logic
from typing import List, Dict


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

########################## THE SCRIPT HAS SOME ISSUES AND IT IS NOT WORKING PROPERLY ##########################
class Translator:
    """
    A class for translating German text to English using a general-purpose LLM (like DeepSeek)
    via prompting and text generation. Handles structured JSON data, text chunking,
    and symbol replacement.
    """
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512, custom_cache_dir: str = None):
        """
        Initializes the Translator with a general-purpose LLM for translation via generation.

        Args:
            model_name (str): Name of the Hugging Face LLM (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B").
            device (str): Device to run the model on ('cuda' or 'cpu').
            max_length (int): Maximum number of new tokens to generate for each translated chunk.
                              This controls the output length of the translation.
            custom_cache_dir (str, optional): Directory to cache Hugging Face models.
        """
        # --- SECURITY IMPROVEMENT: Use environment variable for token ---
        self.access_token = os.environ.get("HF_TOKEN")
        if not self.access_token:
            logging.warning("Hugging Face access token not found in HF_TOKEN environment variable. Proceeding without token.")

        logging.info(f"Initializing LLM model for translation: {model_name} on device: {device}")
        try:
            # --- MODEL LOADING FOR DECODER-ONLY LLM ---
            # Load AutoTokenizer and AutoModelForCausalLM for decoder-only models.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.access_token, cache_dir=custom_cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=self.access_token,
                cache_dir=custom_cache_dir,
                # DeepSeek models often use bfloat16 for efficiency on compatible GPUs
                torch_dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else None,
                # device_map="auto" is crucial for large models to distribute layers across GPUs
                device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
            )
            self.model.eval() # Set model to evaluation mode for inference

            # Get model's internal max input length for chunking
            # This is the maximum number of tokens the model can accept as input.
            self.tokenizer_model_max_length = self.tokenizer.model_max_length 
            if self.tokenizer_model_max_length > 100000 or self.tokenizer_model_max_length <= 0: 
                # Fallback for unrealistic max_length values (some tokenizers report huge values).
                # DeepSeek models typically have a large context window, e.g., 8192 or 4096.
                self.tokenizer_model_max_length = 8192 
                logging.warning(f"Tokenizer model_max_length ({self.tokenizer.model_max_length}) seems unrealistic. Defaulting to {self.tokenizer_model_max_length} for chunking.")
            logging.info(f"LLM model's max input length for tokenization: {self.tokenizer_model_max_length}")

        except Exception as e:
            logging.error(f"Failed to load LLM model or tokenizer {model_name}: {e}")
            logging.error("Please ensure the model is accessible and compatible with AutoModelForCausalLM.")
            raise # Re-raise to stop execution if model loading fails

        self.max_new_tokens_output = max_length # Renamed to reflect output generation length

    def split_text_into_token_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Splits text into chunks based on token limits using the model's tokenizer.
        This helps avoid truncation of very long inputs.

        Args:
            text (str): The input text to split.
            chunk_size (int): The maximum number of tokens per chunk (excluding prompt tokens).

        Returns:
            List[str]: A list of text chunks.
        """
        # Encode the text to get token IDs without special tokens
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        for i in range(0, len(token_ids), chunk_size):
            chunk_token_ids = token_ids[i:i + chunk_size]
            # Decode token IDs back to text for each chunk
            chunks.append(self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True))
        
        return chunks

    def replace_symbols(self, text: str) -> str:
        """Replace '§' with a unique placeholder to prevent translation issues."""
        return text.replace('§', 'Paragraph_SIGN')

    def restore_symbols(self, text: str) -> str:
        """Restore the placeholder 'Paragraph_SIGN' back to '§' after translation."""
        return text.replace('Paragraph_SIGN', '§')


    @lru_cache(maxsize=5000) # Cache up to 5000 unique text translations to avoid re-translating
    def translate_text(self, text: str) -> str:
        """
        Translates a given text snippet from German to English using the LLM via generation.
        Handles text chunking for very long inputs and symbol replacement.
        """
        if not isinstance(text, str) or not text.strip():
            return "" # Return empty string for empty/invalid input

        try:
            # Replace symbols before translation to protect them
            text_with_placeholders = self.replace_symbols(text)
            
            # --- CHUNKING LOGIC FOR LLMs ---
            # LLMs have a maximum input context length. We need to account for the prompt tokens too.
            # Example prompt: "Translate German to English: [GERMAN_TEXT]"
            # Estimate prompt tokens: "Translate German to English: ". This could be ~5-10 tokens.
            # Subtract this from the model's max input length for the actual text chunk.
            estimated_prompt_tokens = 20 # A safe estimate for "Translate German to English: " + some buffer
            effective_chunk_token_limit = self.tokenizer_model_max_length - estimated_prompt_tokens

            if effective_chunk_token_limit <= 0:
                logging.error("Effective chunk token limit is too small. Adjust model_max_length or prompt_tokens_estimate.")
                return text # Cannot proceed with translation


            # Split text into chunks that fit the model's token limit after adding the prompt
            chunks = self.split_text_into_token_chunks(text_with_placeholders, effective_chunk_token_limit)
            
            if len(chunks) > 1:
                logging.info(f"Text too long, splitting into {len(chunks)} chunks for LLM translation.")
            
            translated_chunks = []
            with torch.no_grad(): # Ensure no gradients are computed during inference
                for i, chunk in enumerate(chunks):
                    # --- TRANSLATION PROMPT FOR DEEPSEEK ---
                    # You can experiment with different prompts for better translation quality from LLMs.
                    # E.g., "Übersetze folgenden deutschen Text ins Englische: {chunk}\nEnglisch:"
                    translation_prompt = f"Translate German to English: {chunk}"
                    # For more complex models, you might need chat templates:
                    # messages = [{"role": "user", "content": translation_prompt}]
                    # encoded_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")


                    # Tokenize the prompt and move to model's device
                    inputs = self.tokenizer(translation_prompt, return_tensors="pt", truncation=True, max_length=self.tokenizer_model_max_length).to(self.model.device)

                    # Generate translation
                    generated_output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens_output, # Control output length
                        do_sample=False, # Often use greedy decoding for translation for consistency
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        # temperature=0.1, # Keep low for consistent translation if do_sample=True
                        # top_p=0.9, top_k=0, # If do_sample=True
                        # num_beams=1 # Or >1 for beam search if desired, but can be slow
                    )
                    
                    # Decode the generated output. Slice to get only the newly generated part.
                    # generated_output_ids includes the input prompt tokens.
                    # Slice from the end of the input_ids to get only the generated part.
                    translated_ids = generated_output_ids[0, inputs['input_ids'].shape[1]:]
                    translated_chunk = self.tokenizer.decode(translated_ids, skip_special_tokens=True).strip()
                    translated_chunks.append(translated_chunk)
                    # logging.debug(f"Translated chunk: '{chunk[:50]}...' -> '{translated_chunk[:50]}...'")

            translated_text = " ".join(translated_chunks) # Join translated chunks
            translated_text = self.restore_symbols(translated_text) # Restore '§'
            return translated_text
        
        except Exception as e:
            logging.error(f"Error during translation of text snippet: '{text[:50]}...' -> {e}", exc_info=True) # Log full traceback
            return text # Return original text on failure for robustness


    def recursive_main_content_translate(self, data: List[Dict]) -> List[Dict]:
        """Recursively translates text content in the main content data structure."""
        modified_categories = []
        if not isinstance(data, list):
            logging.warning(f"recursive_main_content_translate: Expected input data to be a list ({type(data)}). Returning empty.")
            return []

        for i, category in enumerate(data):
            # --- Robustness: Validate entry structure ---
            if not isinstance(category, dict) or 'category' not in category or 'paragraphs' not in category:
                logging.warning(f"Skipping invalid category structure at index {i}: {category}. Missing 'category' or 'paragraphs'.")
                continue 
            
            original_category_name = category['category']
            translated_category_name = self.translate_text(original_category_name)
            logging.info(f"Translated category: '{original_category_name}' -> '{translated_category_name}'")

            modified_paragraphs = []
            paragraphs_list = category.get('paragraphs', [])
            if not isinstance(paragraphs_list, list):
                 logging.warning(f"'paragraphs' field in category '{category.get('category')}' is not a list. Skipping its paragraphs.")
                 continue

            for j, paragraph in enumerate(paragraphs_list):
                # --- Robustness: Validate paragraph structure ---
                if not isinstance(paragraph, dict) or 'paragraph' not in paragraph or 'paragraph_id' not in paragraph or 'sentences' not in paragraph:
                    logging.warning(f"Skipping invalid paragraph structure in category '{category.get('category')}' at index {j}: {paragraph}. Missing 'paragraph', 'paragraph_id' or 'sentences'.")
                    continue

                translated_paragraph_text = self.translate_text(paragraph['paragraph'])
                logging.info(f"Translating paragraph: '{paragraph['paragraph_id']}' -> '{translated_paragraph_text[:50]}...'") 
                
                modified_sentences = []
                sentences_list = paragraph.get('sentences', [])
                if not isinstance(sentences_list, list):
                     logging.warning(f"'sentences' field in paragraph '{paragraph.get('paragraph_id')}' is not a list. Skipping its sentences.")
                     continue


                for k, sentence_dict in enumerate(sentences_list):
                    # --- Robustness: Validate sentence structure ---
                    if not isinstance(sentence_dict, dict) or len(sentence_dict) != 1:
                        logging.warning(f"Skipping invalid sentence structure in paragraph '{paragraph.get('paragraph_id')}' at index {k}: {sentence_dict}")
                        continue
                    
                    sen_id, sen_text = list(sentence_dict.items())[0]
                    if not isinstance(sen_text, str): # Ensure sentence text is a string
                        logging.warning(f"Skipping non-string sentence text for ID '{sen_id}' in paragraph '{paragraph.get('paragraph_id')}'. Text: {sen_text}")
                        continue

                    translated_sentence_text = self.translate_text(sen_text)
                    logging.info(f"Translated sentence: '{sen_text[:50]}...' -> '{translated_sentence_text[:50]}...'") 
                    modified_sentences.append({sen_id: translated_sentence_text})

                modified_paragraphs.append({
                    'paragraph': translated_paragraph_text,
                    'paragraph_id': paragraph['paragraph_id'],
                    'sentences': modified_sentences
                })
            modified_categories.append({'category': translated_category_name, 'paragraphs': modified_paragraphs})
        return modified_categories


    def recursive_table_of_contents_translate(self, data: List[Dict]) -> List[Dict]:
        """Recursively translates text content in the table content data structure."""
        modified_table_data = []
        if not isinstance(data, list):
            logging.warning(f"recursive_table_of_contents_translate: Expected input data to be a list ({type(data)}). Returning empty.")
            return []

        for i, table in enumerate(data):
            # --- Robustness: Validate table structure ---
            if not isinstance(table, dict) or 'table_name' not in table or 'table_data' not in table:
                logging.warning(f"Skipping invalid table structure at index {i}: {table}. Missing 'table_name' or 'table_data'.")
                continue

            translated_table_name = self.translate_text(table['table_name'])
            logging.info(f"Translating table name: '{table['table_name']}' -> '{translated_table_name}'")

            modified_sections = []
            sections_list = table.get('table_data', [])
            if not isinstance(sections_list, list):
                 logging.warning(f"'table_data' not a list in table '{table.get('table_name')}'. Skipping its sections.")
                 continue

            for j, section in enumerate(sections_list):
                # --- Robustness: Validate section structure ---
                if not isinstance(section, dict) or 'Section' not in section or 'Section_Id' not in section or 'Content' not in section:
                    logging.warning(f"Skipping invalid section structure in table '{table.get('table_id')}' at index {j}: {section}. Missing 'Section', 'Section_Id' or 'Content'.")
                    continue

                translated_section_name = self.translate_text(section['Section']) 
                logging.info(f"Translating section: '{section.get('Section_Id')}' -> '{translated_section_name[:50]}...'")
                
                modified_entries = []
                content_list = section.get('Content', [])
                if not isinstance(content_list, list):
                     logging.warning(f"'Content' not a list in section '{section.get('Section_Id')}'. Skipping.")
                     continue


                for k, entry in enumerate(content_list):
                    # --- Robustness: Validate entry structure ---
                    if not isinstance(entry, dict) or 'Sign_number' not in entry or 'Sign_name' not in entry or 'description' not in entry:
                         logging.warning(f"Skipping invalid entry structure in section '{section.get('Section_Id')}' at index {k}: {entry}. Missing 'Sign_number', 'Sign_name' or 'description'.")
                         continue
                    
                    # --- IMPROVEMENT: Translate Sign_name more accurately ---
                    original_sign_name = entry.get('Sign_name', '')
                    # Check if original_sign_name is simply 'Zeichen XXX' or 'Sign XXX' (already numbers)
                    if original_sign_name and re.match(r'^(Sign|Zeichen)\s*\d+$', original_sign_name, re.IGNORECASE):
                        translated_sign_name = f"Sign {entry.get('Sign_number', 'Unknown')}" # Reconstruct without translating 'Sign'
                    else:
                        translated_sign_name = self.translate_text(original_sign_name) # Translate actual descriptive name
                    
                    translated_description = self.translate_text(entry.get('description', ''))
                    
                    # Update entry (in-place modification can be ok if data is not reused)
                    # Create a copy to avoid modifying the original 'entry' if it's reused elsewhere
                    modified_entry = entry.copy()
                    modified_entry['Sign_name'] = translated_sign_name
                    modified_entry['description'] = translated_description
                    
                    modified_entries.append(modified_entry) 
                modified_sections.append({
                    "Section": translated_section_name,
                    "Section_Id": section.get('Section_Id'), 
                    "Content": modified_entries
                })
            modified_table_data.append({
                "table_name": translated_table_name,
                "table_id": table.get('table_id'), 
                "table_data": modified_sections
            })
        return modified_table_data
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate German text to English using a DeepSeek LLM')
    # --- MODEL NAME FIX ---
    # Use a DeepSeek model for translation via generation/prompting
    parser.add_argument('--model_name', type=str, default="deepseek-ai/deepseek-llm-7b-base", # Example DeepSeek base model
                        help='Hugging Face LLM model for translation via generation. Example: "deepseek-ai/deepseek-llm-7b-base"')
    
    parser.add_argument('--parsed_main', type=str , default="parsed_main_content_Straßenverkehrs_Ordnung.json", help='Path to the parsed main content JSON file.')
    parser.add_argument('--parsed_table', type=str , default="parsed_table_content_Straßenverkehrs_Ordnung.json", help='Path to the parsed table content JSON file.')
    parser.add_argument('--translated_main', type=str , default="translated_main_content_Straßenverkehrs_Ordnung.json", help='Output filename for translated main content.')
    parser.add_argument('--translated_table', type=str , default="translated_table_content_Straßenverkehrs_Ordnung.json", help='Output filename for translated table content.')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum output length for each translated chunk (new tokens).') # Typically shorter for chunks
    parser.add_argument('--hf_cache_dir', type=str, default="/fast_storage/siddig/driving-license/HF_models", help='Hugging Face cache directory.')
    args = parser.parse_args()

    # --- Initial Logging of Arguments ---
    logging.info("--- Script Arguments ---")
    for arg_name, arg_value in vars(args).items():
        logging.info(f"{arg_name}: {arg_value}")
    logging.info("------------------------")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- Initialize Translator ---
    content_translator = Translator(
        model_name=args.model_name,
        device=device,
        max_length=args.max_length, # This now controls max_new_tokens for generation
        custom_cache_dir=args.hf_cache_dir 
    )

    # --- Load JSON files with proper error handling ---
    main_content_data = []
    table_content_data = []

    try:
        if os.path.exists(args.parsed_main):
            with open(args.parsed_main, 'r', encoding='utf-8') as file:
                main_content_data = json.load(file)
            logging.info(f"Loaded main content from {args.parsed_main}")
        else:
            logging.error(f"Parsed main content file not found: {args.parsed_main}. Skipping translation.")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {args.parsed_main}: {e}")
    except Exception as e:
        logging.error(f"Error loading {args.parsed_main}: {e}")

    try:
        if os.path.exists(args.parsed_table):
            with open(args.parsed_table, 'r', encoding='utf-8') as file:
                table_content_data = json.load(file)
            logging.info(f"Loaded table content from {args.parsed_table}")
        else:
            logging.error(f"Parsed table content file not found: {args.parsed_table}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {args.parsed_table}: {e}")
    except Exception as e:
        logging.error(f"Error loading {args.parsed_table}: {e}")


    # --- Perform Translations and Save Results ---
    if main_content_data:
        logging.info("Translating main content...")
        translated_main_content = content_translator.recursive_main_content_translate(main_content_data)
        safe_model_name_part = args.model_name.replace('/', '_')
        translated_main_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', f'{safe_model_name_part}_{args.translated_main}') 
        
        os.makedirs(os.path.dirname(translated_main_file), exist_ok=True)

        with open(translated_main_file, 'w', encoding='utf-8') as file:
            json.dump(translated_main_content, file, ensure_ascii=False, indent=4)
        logging.info(f"Translated main content saved to {translated_main_file}")
    else:
        logging.warning("Main content data is empty, skipping translation.")


    if table_content_data:
        logging.info("Translating table content...")
        translated_table_content = content_translator.recursive_table_of_contents_translate(table_content_data)
        safe_model_name_part = args.model_name.replace('/', '_')
        translated_table_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', f'{safe_model_name_part}_{args.translated_table}') 

        os.makedirs(os.path.dirname(translated_table_file), exist_ok=True)

        with open(translated_table_file, 'w', encoding='utf-8') as file:
            json.dump(translated_table_content, file, ensure_ascii=False, indent=4)
        logging.info(f"Translated table content saved to {translated_table_file}")
    else:
        logging.warning("Table content data is empty, skipping translation.")