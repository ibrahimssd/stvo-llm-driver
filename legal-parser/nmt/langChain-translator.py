import json
import logging
import argparse
import asyncio
import os
from langchain_community.document_transformers import DoctranTextTranslator
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from .env file, if it exists
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

class LangchainTranslator:
    def __init__(self, openai_api_key: str):
        """
        Initializes the LangchainTranslator with the OpenAI API key.

        Args:
            openai_api_key (str): Your OpenAI API key.
        """
        self.openai_api_model = os.
            
        if not self.openai_api_model:
                raise ValueError("OPENAI_API_MODEL environment variable not set. Please set it or pass the model name directly.")

        self.translator = DoctranTextTranslator(
                language="german",
                openai_api_key=self.openai_api_key,
                openai_api_model=self.openai_api_model,
        )


    async def translate_text(self, text: str) -> str:
        documents = [Document(page_content=text)]
        translated_documents = await self.translator.atransform_documents(documents)
        return translated_documents[0].page_content

    async def recursive_main_content_translate(self, data: list) -> list:
        modified_categories = []
        for category in data:
            translated_category = await self.translate_text(category['category'])
            logging.info(f"Translated category: {translated_category}")

            modified_paragraphs = []
            for paragraph in category['paragraphs']:
                translated_paragraph = await self.translate_text(paragraph['paragraph'])
                logging.info(f"Translated paragraph: {translated_paragraph}")

                modified_sentences = []
                for sentence_dict in paragraph['sentences']:
                    modified_sentence_dict = {}
                    for sen_id, sen_text in sentence_dict.items():
                        translated_text = await self.translate_text(sen_text)
                        logging.info(f"Translated sentence [{sen_id}]: {translated_text}")
                        modified_sentence_dict[sen_id] = translated_text
                    modified_sentences.append(modified_sentence_dict)

                modified_paragraphs.append({
                    'paragraph': translated_paragraph,
                    'paragraph_id': paragraph['paragraph_id'],
                    'sentences': modified_sentences
                })
            modified_categories.append({'category': translated_category, 'paragraphs': modified_paragraphs})
        return modified_categories

    async def recursive_table_of_contents_translate(self, data: list) -> list:
        modified_table_data = []
        for table in data:
            translated_table_name = await self.translate_text(table['table_name'])
            modified_sections = []
            for section in table['table_data']:
                translated_section = await self.translate_text(section['Section'])
                modified_entries = []
                for entry in section['Content']:
                    translated_sign_name = f"Sign {entry['Sign_number']}"
                    translated_description = await self.translate_text(entry['description'])
                    modified_entries.append({
                        'Sign_name': translated_sign_name,
                        'Sign_number': entry['Sign_number'],
                        'description': translated_description
                    })
                modified_sections.append({
                    "Section": translated_section,
                    "Section_Id": section['Section_Id'],
                    "Content": modified_entries
                })
            modified_table_data.append({
                "table_name": translated_table_name,
                "table_id": table['table_id'],
                "table_data": modified_sections
            })
        return modified_table_data

async def main(args):
    for arg in vars(args):
        logging.info("%s: %s", arg, getattr(args, arg))

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    content_translator = LangchainTranslator(openai_api_key=openai_api_key)

    with open(args.parsed_main, 'r', encoding='utf-8') as file:
        main_content_data = json.load(file)

    with open(args.parsed_table, 'r', encoding='utf-8') as file:
        table_content_data = json.load(file)

    translated_main_content = await content_translator.recursive_main_content_translate(main_content_data)
    translated_main_file = os.path.join("../../data", args.translated_main)
    with open(translated_main_file, 'w', encoding='utf-8') as file:
        json.dump(translated_main_content, file, ensure_ascii=False, indent=4)

    translated_table_content = await content_translator.recursive_table_of_contents_translate(table_content_data)
    translated_table_file = os.path.join("../../data", args.translated_table)
    with open(translated_table_file, 'w', encoding='utf-8') as file:
        json.dump(translated_table_content, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate German legal text to English using Langchain')
    parser.add_argument('--parsed_main', type=str, default="parsed_main_content_Straßenverkehrs_Ordnung.json",
                        help='Path to the parsed main content JSON')
    parser.add_argument('--parsed_table', type=str, default="parsed_table_content_Straßenverkehrs_Ordnung.json",
                        help='Path to the parsed table content JSON')
    parser.add_argument('--translated_main', type=str, default="main_translated_Straßenverkehrs_Ordnung.json",
                        help='Filename for the translated main content')
    parser.add_argument('--translated_table', type=str, default="table_translated_Straßenverkehrs_Ordnung.json",
                        help='Filename for the translated table content')
    args = parser.parse_args()

    asyncio.run(main(args))
