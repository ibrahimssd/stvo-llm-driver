import requests
from bs4 import BeautifulSoup
import json
import os
import time
from collections import Counter
import re
from citator import ContentCitator
import argparse


class MainContentExtractor(ContentCitator):
    def __init__(self, url, base_url,save_image_dir):
        self.url = url
        self.base_url = base_url
        self.save_image_dir = save_image_dir

    def fetch_html(self, retry_count=5, delay=3, timeout_duration=2000):
        for attempt in range(retry_count):
            try:
                response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=timeout_duration)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                if attempt < retry_count - 1:
                    time.sleep(delay)
                    continue
                else:
                    raise SystemExit(f"Failed to fetch data after {retry_count} attempts due to: {e}")
            
    def get_html_content(self):
        response = requests.get(self.url)
        response.raise_for_status()  # Ensure the response status is OK
        return response.content

    def download_image(self,image_url, save_dir, retries=10, delay=5):
        os.makedirs(save_dir, exist_ok=True)
        filename = image_url.split('/')[-1]
        filepath = os.path.join(save_dir, filename)
        print(f"Attempting to download image: {image_url}")  # Debug information

        for attempt in range(retries):
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                response = requests.get(image_url, headers=headers, stream=True, timeout=200)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    return filename
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1}: Failed to download image {image_url} due to: {e}")
                time.sleep(delay)
        print(f"Failed to download image after {retries} attempts: {image_url}")  # Log failed download
        return "Failed image download"

    def extract_main_content(self,html):
        soup = BeautifulSoup(html, 'html.parser')
        data = []
        failed_images = []
        image_ids = []

        current_category_name = None
        current_category_sections = []

        for section in soup.find_all('div', class_='jnnorm'):
            new_category = section.find_previous('h2')
            if new_category:
                category_name = new_category.get_text(strip=True)
                print(f"Extracting data from category: {category_name}")
                if category_name != current_category_name:
                    if current_category_sections:
                        data.append({
                            'category': current_category_name,
                            'paragraphs': current_category_sections
                        })
                    current_category_name = category_name
                    current_category_sections = []  # Reset for new category

            h3 = section.find('h3')
            section_title = h3.get_text(strip=True) if h3 else "No Paragraph Title"


            if section_title == "Schlussformel" or section_title.startswith("Anlage") :
                break
            
            sentences = []
            sentence = {}
            last_paragraph_id = None
            content_accumulator = []

            for element in section.descendants:
                if element.name == 'div' and 'jurAbsatz' in element.get('class', []):
                    text_content = element.get_text(strip=True)
                    parts = text_content.split()
                    # Handle new paragraph IDs or accumulate sentences as a single paragraph
                    if parts and parts[0].endswith(')') and parts[0].startswith('('):
                        if content_accumulator:  # Store previous content if exists
                            sentence[last_paragraph_id] = ' '.join(content_accumulator)
                            sentences.append(sentence)
                            sentence = {}
                            content_accumulator = []
                        last_paragraph_id = parts[0]
                        # convert into a valid paragraph ID
                        last_paragraph_id = last_paragraph_id[1:-1]
                        
                    elif not last_paragraph_id:  # If no paragraph ID, create a generic one
                        last_paragraph_id = 1
                    content_accumulator.append(text_content)

                elif element.name == 'img':
                    image_url = requests.compat.urljoin(self.base_url, element['src'])
                    image_id = self.download_image(image_url, self.save_image_dir)
                    if image_id:
                        image_ids.append(image_id)
                        image_ref = f"<img>{image_id}</img>"
                        content_accumulator.append(image_ref)
                    else:
                        failed_images.append(image_url)
                        content_accumulator.append("[Image unavailable]")

            # Store the last piece of accumulated content
            if content_accumulator and last_paragraph_id:
                sentence= {}
                sentence[last_paragraph_id] = ' '.join(content_accumulator)
                sentences.append(sentence)

            # Handle section ID extraction from the section title
            para_ID = re.search(r'§\s*\d+[a-z]?', section_title)
            if sentence:
                current_category_sections.append({
                    'paragraph': section_title,
                    'paragraph_id': para_ID.group() if para_ID else "No Paragraph ID",
                    'sentences': sentences
                })

        # Append the last category's data
        if current_category_sections:
            data.append({
                'category': current_category_name,
                'paragraphs': current_category_sections
            })
        
        # remove null categories Kategorie": null,
        data = [category for category in data if category['category']]
        
        return data, failed_images, image_ids




    def check_missing_images(self,image_ids):
        print("image_ids length: ", len(image_ids))
        print("image_ids duplicates: ", self.find_duplicates(image_ids))
        
        image_ids = set(image_ids)
        # remove file extension from image ids
        image_ids = set([img_id.split('.')[0] for img_id in image_ids])
        image_files = set([filename.split('.')[0] for filename in os.listdir(self.save_image_dir) if filename.endswith('.jpg')])
        missing_images = image_ids - image_files
        if missing_images:
            print(f"Missing images: {missing_images}")
        else:
            # print length of image_ids and image_files
            print(f"Image IDs: {len(image_ids)}")
            print(f"Image Files: {len(image_files)}")
            print("No missing images. All images referenced are downloaded.")

    def find_duplicates(self,input_list):
        counter = Counter(input_list)
        return [item for item, count in counter.items() if count > 1]


    ############### Reference signs in the extracted rules ####################
    def refrence_extracted_rules_with_signs(self,data):
        modified_categories = []  # This will store the modified categories
        for category in data:
            modified_paragraphs = []  # This will store modified sections for the current category
            for section in category['paragraphs']:  # Adjust if the actual key is different
                modified_sentences =[]
                for sentence_dict in section['sentences']:  # Adjust if the actual key is different
                    modified_sentence = {}
                    for sen_id, sen_text in sentence_dict.items():
                        modified_text = self.reference_sign(sen_text)
                        modified_sentence[sen_id] = modified_text
                    modified_sentences.append(modified_sentence)
                modified_paragraph = {
                    'paragraph': section['paragraph'],  # Adjust if the actual key is different
                    'paragraph_id': section['paragraph_id'],  # Adjust if the actual key is different
                    'sentences': modified_sentences
                }
                modified_paragraphs.append(modified_paragraph)
            modified_categories.append({
                'category': category['category'],  # Adjust if the actual key is different
                'paragraphs': modified_paragraphs
            })
        
        # save_data(modified_categories, filename)  # Save the modified categories instead of the original data

        return modified_categories


    ########################## Reference paragraphs and sentences ############################

    def refrence_extracted_rules_with_paragraphs_and_sentences(self,data):
    
        modified_categories = []  # This will store the modified categories
        # extraxt paragraphs ids from the data
        sentence_ids = []
        for category in data:
            for section in category['paragraphs']:
                for sentence_dict in section['sentences']:
                    for sen_id, sen_text in sentence_dict.items():
                        sentence_ids.append(sen_id)
        # print(sentence_ids)
        for category in data:
            modified_paragraphs = []  # This will store modified sections for the current category
            for section in category['paragraphs']:  # Adjust if the actual key is different
                modified_sentences = []
                for sentence_dict in section['sentences']:  # Adjust if the actual key is different
                    modified_sentence = {}
                    for sen_id, sen_text in sentence_dict.items():
                        modified_text = self.reference_paragraphs_and_sentences(sen_text, section['paragraph_id'])
                        modified_sentence[sen_id] = modified_text
                    modified_sentences.append(modified_sentence)
                modified_paragraph = {
                    'paragraph': section['paragraph'],  # Adjust if the actual key is different
                    'paragraph_id': section['paragraph_id'],  # Adjust if the actual key is different
                    'sentences': modified_sentences
                }
                modified_paragraphs.append(modified_paragraph)
            modified_categories.append({
                'category': category['category'],  # Adjust if the actual key is different
                'paragraphs': modified_paragraphs
            })
        # save_data(modified_categories, filename)  # Save the modified categories instead of the original data

        return modified_categories

    ####################### Reference Anlagen and Abschnitte ############################

    def refrence_extracted_rules_with_anlage_and_abschnitte(self,data):
    
        modified_categories = []  # Store the modified categories
        for category in data:
            
            modified_paragraphs = []
            for section in category['paragraphs']:  # Assuming structure has 'paragraphs' within 'category'
                modified_sentences = []
                for sentence_dict in section['sentences']:  # Assuming each section contains 'sentences' which is a list of dicts
                    for sen_id, sen_text in sentence_dict.items():
                        modified_text = self.refrence_tables_and_sections(sen_text)
                        modified_sentences.append({sen_id: modified_text})
                modified_paragraph = {
                    'paragraph': section['paragraph'],  # Assuming 'paragraph' and 'paragraph_id' keys are present
                    'paragraph_id': section['paragraph_id'],
                    'sentences': modified_sentences
                }
                modified_paragraphs.append(modified_paragraph)
            modified_categories.append({
                'category': category['category'],  # Assuming 'category' key is present in the data structure
                'paragraphs': modified_paragraphs
            })

        # save_data(modified_categories, filename)  # Save the modified categories instead of the original data

        return modified_categories
    

    

    def save_data(self,data, filename=None):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)






    def class_main(self):
        
        # html = fetch_html(url)
        html = self.get_html_content()
        structured_data, failed_images , image_ids = self.extract_main_content(html)
        try:
            
                image_count = len([filename for filename in os.listdir(self.save_image_dir) if filename.endswith('.jpg')])
                print(f"Downloaded {image_count} images.")

                print("Failed images: ", failed_images if failed_images else "No failed images.")
                
                # Check for missing images
                self.check_missing_images(image_ids)

        except FileNotFoundError: 
            print("No images downloaded. Image Folder is empty.")

        return structured_data

    
if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.gesetze-im-internet.de/stvo_2013/BJNR036710013.html", help="url to extract data")
    parser.add_argument("--base_url", default = "https://www.gesetze-im-internet.de", help="base url")
    parser.add_argument("--save_image_dir", default="main_content_Straßenverkehrs_Ordnung_images", help="directory to save images")
    parser.add_argument("--save_data_dir", default="parsed_main_content_Straßenverkehrs_Ordnung.json", help="directory to save parsed data")
    args = parser.parse_args()

    
    
    main_content_extractor = MainContentExtractor(args.url, args.base_url, args.save_image_dir)
    extracted_data= main_content_extractor.class_main()
    
    # save the extracted data for parser evaluation
    main_content_extractor.save_data(extracted_data, filename=f'./parser-eval/{args.save_data_dir}')

    # Reference the extracted rules
    parsed_data = main_content_extractor.refrence_extracted_rules_with_signs(extracted_data)

    # Reference the extracted rules with paragraphs and sentences
    parsed_data= main_content_extractor.refrence_extracted_rules_with_paragraphs_and_sentences(parsed_data)

    # Reference the extracted rules with Anlagen and Abschnitte
    parsed_data = main_content_extractor.refrence_extracted_rules_with_anlage_and_abschnitte(parsed_data)
    
    #save the data
    main_content_extractor.save_data(parsed_data, filename=f'./nmt/{args.save_data_dir}')
    print("Data has been parsed and saved.")
    
    


 