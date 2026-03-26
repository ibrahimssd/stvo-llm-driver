import requests
from bs4 import BeautifulSoup
import os
import json
import time
from collections import Counter
import re
from citator import ContentCitator
import argparse


class TableContentExtractor(ContentCitator):
    def __init__(self, url,base_url,save_image_dir):
        self.url = url
        self.base_url = base_url 
        self.save_image_dir = save_image_dir
    
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

    def extract_table_content(self,html):
        soup = BeautifulSoup(html, 'html.parser')
        data = []
        image_ids = []
        # Locate the start of the relevant section after "Schlussformel"
        # section_title.startswith("Anlage") 
        schlussformel = soup.find('span', string=lambda text: text and "Schlussformel" in text)
        Anlage = soup.find('span', string=lambda text: text and "Anlage" in text)
        
        
        
        if schlussformel:
            print("Schlussformel HTML:", schlussformel)
            tables = schlussformel.find_all_next('table')
            print("Tables found after Schlussformel:", len(tables))
        elif Anlage:
            print("Anlage HTML:", Anlage)
            tables = Anlage.find_all_next('table')
            print("Tables found after Anlage:", len(tables))
        else:
            print("No Schlussformel or Anlage found")

                
        for table in tables:
                table_name = table.find_previous('h3').get_text(strip=True) if table.find_previous('h3') else "No Table Name"
                print(f"Extracting data from table: {table_name}")
                table_content = []
                current_section = None

                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                if not headers:
                    headers = ['Column ' + str(i + 1) for i in range(len(table.find('tr').find_all('td')))]

                for row in table.find_all('tr'):
                    cells = row.find_all('td')
                    row_data = {}

                    for index, cell in enumerate(cells):
                        text_parts = list(cell.stripped_strings)
                        img_tag = cell.find('img')
                        if img_tag:
                            image_url = requests.compat.urljoin(self.base_url, img_tag['src'])
                            image_id = self.download_image(image_url,self.save_image_dir)
                            image_ids.append(image_id)
                            text_above = text_parts[0] if text_parts else ""
                            text_below = text_parts[-1] if len(text_parts) > 1 else ""
                            # cell_data = f"{text_above} : [Image ID: {image_id}] : {text_below}"
                            cell_data = f"{text_above} : {text_below}"
                        else:
                            cell_data = ' '.join(text_parts)

                        if index < len(headers):
                            row_data[headers[index]] = cell_data
                        else:
                            row_data["Extra Data"] = cell_data

                    if len(row_data) == 1:
                        # Save previous section if exists
                        if current_section:
                            table_content.append(current_section)
                        # Start new section
                        section_info = list(row_data.values())[0]
                        abschnitt_pattern = re.compile(r'Abschnitt\s+\d+')
                        abschnitt_match = abschnitt_pattern.search(section_info)
                        if abschnitt_match:
                            section_Id = abschnitt_match.group()
                            section_Id = section_Id.split()[-1]
                            section_Id = int(section_Id)
                            section_Id = str(section_Id)
                        else:
                            section_Id = ""

                        current_section = {"Section": section_info,
                                            "Section_Id": section_Id,
                                            "Content": []}
                    else:
                        # It's a regular row with image information, add to current section
                        if current_section:
                            # rename the keys of the row_data dictionary as image_id, image_info, explanation
                            items = list(row_data.values())
                            try :
                                sign_id = items[0]
                                image_info = items[1]
                                explanation = items[2]
                            except:
                                sign_id = ""
                                image_info = items[0]
                                explanation = items[1]
                                
                            
                            # extraxt the image_id from the image_info *(float) -> 1.1
                            # zeichen_pattern = re.compile(r'Zeichen\s+\d+')
                            zeichen_pattern = re.compile(r'Zeichen\s+\d+(?:\.\d+)?')

                            zeichen_match = zeichen_pattern.search(image_info)
                            if zeichen_match:
                                img_number = zeichen_match.group()
                                img_number = img_number.split()[-1]
                                img_number = str(img_number)

                            else:
                                img_number = ""
                            
                            # there is no "Zeichnen_id": "", "Zeichennummer": "", "Zeichen": "", skip the row
                            if sign_id == "" and img_number == "" and image_info == "":
                                continue

                            # extract Zechen d from image info and dlete it from the image_info
                            sign_name , image_info = self.extract_zeichen(image_info)
                            
                            current_section['Content'].append({ "Sign_id": sign_id,
                                                            "Sign_number": img_number,
                                                            "Sign_name": sign_name,
                                                            "image": f'<img>{image_id}</img>',
                                                            "description": image_info + " " + explanation,
                                                            })
                            
                            # current_section['content'].append(row_data)

                # Append the last section after processing all rows
                if current_section:
                    table_content.append(current_section)

                # Append table data
                anlage_pattern = re.compile(r'Anlage\s+\d+')
                anlage_match = anlage_pattern.search(table_name)
                if anlage_match:
                    table_Id = anlage_match.group()
                    table_Id = table_Id.split()[-1]
                    table_Id = int(table_Id)
                    table_Id= str(table_Id)
                data.append({
                    "table_name": table_name,
                    "table_id": table_Id,
                    "table_data": table_content
                })
        
        
        # post process the data concatenate 
        return data , image_ids

    def extract_zeichen(self,text):
        # Define the regular expression pattern for "Zeichen" followed by a number in float or integer format
        pattern = r"(Zeichen\s+\d+(?:\.\d+)?)"
        # pattern = r"(Zeichen\s\d+)"
        
        # Search for the pattern in the text
        match = re.search(pattern, text)
        
        if match:
            # Extract the matched "Zeichen + number"
            extracted = match.group(0)
            
            # Remove the extracted part from the original text
            modified_text = re.sub(pattern, '', text).strip()
            
            return extracted, modified_text
        else:
            return "", text
    
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

        image_ids = set(image_ids)
        # remove file extension from image ids
        image_ids = [re.sub(r'\.\w+$', '', image_id) for image_id in image_ids]
        print("image_ids length after removing file extension: ", len(image_ids))
        print("image_ids duplicates after removing file extension: ", self.find_duplicates(image_ids))


    def find_duplicates(self,input_list):
        counter = Counter(input_list)
        return [item for item, count in counter.items() if count > 1]
    
    # def clean_and_aggregate_entries(self, data):
    #     for table in data:
    #         previous_section = None  # Initialize variable to keep track of the previous section

    #         for index, section in enumerate(table['table_data']):
    #             # iterate over section content if sign_id , sign_number and sign_name , merge the content item with the previous one
    #             for i, entry in enumerate(section['Content']):
    #                 if entry['Sign_id'] == "" and entry['Sign_number'] == "" and entry['Sign_name'] == "":
    #                         section['Content'][i-1]['description'] += " " + entry['description']
                        

    #             # Check if the section starts with "Abschnitt"
    #             if not section['Section'].startswith("Abschnitt"):
    #                 if previous_section is not None:
    #                     # Merge current section's content into the previous section
    #                     previous_section['Content'] += section['Content']
    #                     # Add the section title to the previous section's description last line
    #                     previous_section['Content'][-1]['description'] += f" {section['Section']}"
                        
    #                     # Mark current section for removal since it has been merged
    #                     table['table_data'][index] = None
    #             else:
    #                 # Update the previous section to the current one as it starts with "Abschnitt"
    #                 previous_section = section

    #         # Remove None entries (merged sections) from the table data
    #         table['table_data'] = [section for section in table['table_data'] if section is not None]

    #     return data
    def clean_and_aggregate_entries(self, data):
        for table in data:
            new_table_data = []
            previous_section = None
            
            for section in table['table_data']:
                if not section.get('Content'):
                    continue

                # Merge descriptions for items without identifiers into the last or upcoming valid content
                new_content = []
                last_description = ""
                for entry in section['Content']:
                    if all(entry.get(key) == "" for key in ['Sign_id', 'Sign_number', 'Sign_name']):
                        last_description += " " + entry['description'].strip()
                    else:
                        entry['description'] = last_description.strip() + " " + entry['description'].strip()
                        last_description = ""  # Reset after appending
                        new_content.append(entry)
                
                section['Content'] = new_content

                # Handle section merging based on title
                if section['Section'].startswith("Abschnitt"):
                    if previous_section:
                        new_table_data.append(previous_section)
                    previous_section = section
                else:
                    if previous_section:
                        previous_section['Content'] += section['Content']

            if previous_section:
                new_table_data.append(previous_section)
            
            table['table_data'] = new_table_data

        return data



    ############### Reference signs in the extracted rules ####################
    def refrence_table_rules_with_signs(self,data):
        # iterate over the data descrition and apply the reference_sign function to the description
        modified_table_data = []
        for table in data:
            modified_tabel_name = self.reference_sign(table['table_name'])
            modified_sections = []
            for section in table['table_data']:
                modified_section = self.reference_sign(section['Section'])
                modified_entries = []
                for entry in section['Content']:
                    modified_description = self.reference_sign(entry['description'])
                    entry['description'] = modified_description
                    modified_entries.append(entry)
                modified_sections.append({
                    "Section": modified_section,
                    "Section_Id": section['Section_Id'],
                    "Content": modified_entries
                })
            modified_table_data.append({
                "table_name": modified_tabel_name,
                "table_id": table['table_id'],
                "table_data": modified_sections
            })
        
        return modified_table_data


    ############### Reference paragraphs and sentences in the extracted rules ####################
    def refrence_table_rules_with_paragraphs_and_sentences(self,data):
        # iterate over the data descrition and apply the reference_paragraphs_and_sentences function to the description
        modified_table_data = []
        for table in data:
            modified_table_name = self.reference_paragraphs_and_sentences(table['table_name'])
            modified_sections = []
            for section in table['table_data']:
                modified_section = self.reference_paragraphs_and_sentences(section['Section'])
                modified_entries = []
                for entry in section['Content']:
                    modified_description = self.reference_paragraphs_and_sentences(entry['description'])
                    entry['description'] = modified_description
                    modified_entries.append(entry)
                modified_sections.append({
                    "Section": modified_section,
                    "Section_Id": section['Section_Id'],
                    "Content": modified_entries
                })
            modified_table_data.append({
                "table_name": modified_table_name,
                "table_id": table['table_id'],
                "table_data": modified_sections
            })
        
        return modified_table_data
        

    ################### Reference Anlagen and Abschnitte in the extracted rules ####################
    def refrence_table_rules_with_tabels_and_sections(self,data):
        # iterate over the data descrition and apply the tag_anlagen_and_abschnitte function to the description
        modified_table_data = []
        for table in data:
            # modified_table_name = self.refrence_tables_and_sections(table['table_name'])
            modified_sections = []
            for section in table['table_data']:
                # modified_section = self.refrence_tables_and_sections(section['Section'])
                modified_entries = []
                for entry in section['Content']:
                    modified_description = self.refrence_tables_and_sections(entry['description'])
                    entry['description'] = modified_description
                    modified_entries.append(entry)
                modified_sections.append({
                    "Section": section['Section'],
                    "Section_Id": section['Section_Id'],
                    "Content": modified_entries
                })
            modified_table_data.append({
                "table_name": table['table_name'],
                "table_id": table['table_id'],
                "table_data": modified_sections
            })
        
        return modified_table_data

    
    def save_data(self,data, filename=None):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)



    def class_main(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            content = response.text
            extracted_data , image_ids = self.extract_table_content(content)
            extracted_data = self.clean_and_aggregate_entries(extracted_data)
            
            # Check for missing images
            self.check_missing_images(image_ids)

        return extracted_data


if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.gesetze-im-internet.de/stvo_2013/BJNR036710013.html", help="url to extract data")
    parser.add_argument("--base_url", default = "https://www.gesetze-im-internet.de", help="base url")
    parser.add_argument("--save_image_dir", default='table_content_Straßenverkehrs_Ordnung_images', help="directory to save images")
    parser.add_argument("--save_data_dir", default='parsed_table_content_Straßenverkehrs_Ordnung.json', help="directory to save parsed data")
    args = parser.parse_args()
    



    table_content_extractor = TableContentExtractor(args.url,args.base_url,args.save_image_dir)
    extracted_data =   table_content_extractor.class_main()

    # save the extracted data
    table_content_extractor.save_data(extracted_data, filename=f'./parser-eval/{args.save_data_dir}')

    # Reference the extracted data with signs
    parsed_data = table_content_extractor.refrence_table_rules_with_signs(extracted_data)

    # Reference the extracted data with paragraphs and sentences
    parsed_data = table_content_extractor.refrence_table_rules_with_paragraphs_and_sentences(parsed_data)

    # Reference the extracted data with Anlagen and Abschnitte
    parsed_data = table_content_extractor.refrence_table_rules_with_tabels_and_sections(parsed_data)

    # Save the data
    table_content_extractor.save_data(parsed_data, filename=f'./nmt/{args.save_data_dir}')
    print("Data has been parsed and saved.")




