
import json
from transformers import pipeline
import torch
from functools import lru_cache
import logging
import argparse
import re


# Setup logging
logging.basicConfig(level=logging.INFO)


################ THIS SCRIPT HAS BEEN TESTED WITH THE FOLLOWING MODEL ADND IT IS WORKING FINE ################
class Translator:
    def __init__(self, model_name, device="cuda", max_length=512, custom_cache_dir=None):
        self.access_token = "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"
        self.translator = pipeline("translation_de_to_en",
                                   model=model_name,
                                   token=self.access_token,
                                    # cache_dir=custom_cache_dir,  # Set the custom cache directory
                                    device=device)
        self.max_length = max_length

    def split_text(self,text):
        """
        Split text into chunks, each with a maximum length of `max_length`.
        This simplistic splitter cuts on space characters.
        """
        words = text.split()
        current_chunk = []
        chunks = []

        for word in words:
            if sum(len(w) + 1 for w in current_chunk + [word]) > self.max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def replace_symbols(self,text):
        # Replace '§' with a unique placeholder
        # "I.Allgemeine Verkehrsregeln"
        return text.replace('§', 'Paragraph_SIGN')

    def restore_symbols(self,text):
        # Restore 'SECTION_SIGN' back to '§'
        return text.replace('Paragraph_SIGN', '§')


    @lru_cache(maxsize=5000)
    def translate_text(self,text):
        try:
            # Handle splitting text to ensure it fits the model's max length
            text_with_placeholders  = self.replace_symbols(text)
            
            chunks = self.split_text(text_with_placeholders)
            translated_chunks = [ self.translator(chunk, max_length=self.max_length)[0]['translation_text']  for chunk in chunks]
            # translated_chunks = [translator.translate(chunk) for chunk in chunks]
            translated_text = " ".join(translated_chunks)

            translated_text = self.restore_symbols(translated_text)
            return translated_text
        except Exception as e:
            logging.error(f"Error while translating text: {e}")
            return text

    # Add this new method to your Translator class
    def translate_paragraph_title(self, title_text):
        """
        Intelligently translates paragraph titles like "§ 8 Vorfahrt"
        by separating the marker from the text.
        """
        # Regex to find a pattern like "§ 8 " and capture it, and also capture the text part.
        match = re.match(r"(§\s*\d+\s*)(.*)", title_text)
        
        if match:
            marker = match.group(1)  # This will be "§ 8 "
            text_to_translate = match.group(2).strip() # This will be "Vorfahrt"
            
            if not text_to_translate:
                return marker # Return just the marker if there's no text
                
            # Translate only the actual text part
            translated_text = self.translate_text(text_to_translate)
            
            # Recombine them
            return f"{marker}{translated_text}"
        else:
            # If the title doesn't match the pattern, translate the whole thing
            return self.translate_text(title_text)
        
    # Replace your existing recursive_main_content_translate with this version
    def recursive_main_content_translate(self, data):
        modified_categories = []
        for category in data:
            translated_category = self.translate_text(category['category'])
            logging.info(f"Translated category: {translated_category}")

            modified_paragraphs = []
            for paragraph in category['paragraphs']:
                # --- KEY CHANGE: Use the new specific function for the paragraph title ---
                translated_paragraph = self.translate_paragraph_title(paragraph['paragraph'])
                logging.info(f"Translating paragraph title: {paragraph['paragraph']} -> {translated_paragraph}")
                
                modified_sentences = []
                for sentence_dict in paragraph['sentences']:
                    for sen_id, sen_text in sentence_dict.items():
                        # Use the standard translator for full sentences
                        translated_text = self.translate_text(sen_text)
                        # logging.info(f"Translating sentence: {translated_text}") # Optional: can be noisy
                        modified_sentences.append({sen_id: translated_text})
                
                modified_paragraphs.append({
                    'paragraph': translated_paragraph,
                    'paragraph_id': paragraph['paragraph_id'],
                    'sentences': modified_sentences
                })
            modified_categories.append({'category': translated_category, 'paragraphs': modified_paragraphs})
        
        return modified_categories


    def recursive_table_of_contents_translate(self,data):
            modified_table_data = []
            for table in data:
                translated_table_name = self.translate_text(table['table_name'])
                modified_sections = []
                for section in table['table_data']:
                    translatd_section = self.translate_text(section['Section'])  
                    modified_entries = []
                    for entry in section['Content']:
                        # translate Sign_name -> Sign + number
                        translated_sign_name = "Sign " + entry['Sign_number']
                        entry['Sign_name'] = translated_sign_name
                        translated_description = self.translate_text(entry['description'])
                        entry['description'] = translated_description
                        modified_entries.append(entry)
                    modified_sections.append({
                        "Section": translatd_section,
                        "Section_Id": section['Section_Id'],
                        "Content": modified_entries
                    })
                modified_table_data.append({
                    "table_name": translated_table_name,
                    "table_id": table['table_id'],
                    "table_data": modified_sections
                })
            
            return modified_table_data
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate German text to English')
    parser.add_argument('--model_name', type=str, default="facebook/m2m100_418M", help='Model for translation.')
    parser.add_argument('--parsed_main', type=str , default=None, help='Directory to save the parsed main content')
    parser.add_argument('--parsed_table', type=str , default=None, help='Directory to save the parsed table content')
    parser.add_argument('--translated_main', type=str , default="translated_main_content_Straßenverkehrs_Ordnung.json", help='Directory to save the translated main content')
    parser.add_argument('--translated_table', type=str , default="translated_table_content_Straßenverkehrs_Ordnung.json", help='Directory to save the translated table content')
    parser.add_argument('--max_length', type=int, default=400, help='Maximum length of text chunks.')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for translated files.')
    args = parser.parse_args()

    for arg in vars(args):
        logging.info("%s: %s", arg, getattr(args, arg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Translator(model_name=args.model_name, device=device)
    content_translator = Translator(model_name=args.model_name, device=device, max_length= args.max_length, 
                                    custom_cache_dir="/fast_storage/siddig/driving-license/HF_models")
    # Load the JSON file
    if args.parsed_main :
        with open(args.parsed_main, 'r', encoding='utf-8') as file:
            main_content_data = json.load(file)

        # save the translated data to a new JSON file
        translated_main_content = content_translator.recursive_main_content_translate(main_content_data)
        model_name = args.model_name.split('/')[1]
        translated_main_file = f'{args.out_dir}/{model_name}_{args.translated_main}'
        with open(translated_main_file, 'w', encoding='utf-8') as file:
            json.dump(translated_main_content, file, ensure_ascii=False, indent=4)


    if args.parsed_table:
        # translate table of contents
        with open(args.parsed_table, 'r', encoding='utf-8') as file:
            table_content_data = json.load(file)


        # save carfully the translated data to a new JSON file consider non-ascii characters
        translated_table_content = content_translator.recursive_table_of_contents_translate(table_content_data)
        model_name = args.model_name.split('/')[1]
        translated_table_file = f'{args.out_dir}/{model_name}_{args.translated_table}'
        with open(translated_table_file, 'w', encoding='utf-8') as file:
            json.dump(translated_table_content, file, ensure_ascii=False, indent=4)







