import re
import json
from collections import Counter




class ContentCitator:

    def __init__(self, data):
        self.data = data
            

    # #########################################  [1] ZEIHCEN  #########################################

    def reference_sign(self,text):
        # Define the regex pattern to capture Zeichen references, including ranges and lists
        pattern = re.compile(
            r'\bZeichen\s+'  # Start with the word 'Zeichen'
            r'((\d{1,6}(\.\d)?)(\s*(?:bis|und|oder|,)\s*\d{1,6}(\.\d)?)*'  # Match numbers, potential decimals and ranges or lists
            r')', 
            re.IGNORECASE
        )

        def replace_sign(match):
            # Split the match to handle individual numbers or ranges
            numbers = match.group(1).replace(' und ', ', ').replace(' oder ', ', ').replace(' bis ', ' to ').split(',')
            formatted_numbers = []
            for number in numbers:
                if ' to ' in number:
                    # Expand range expressions into individual numbers
                    start, end = map(int, number.split(' to '))
                    formatted_numbers.extend(f"<sign>{i}</sign>" for i in range(start, end + 1))
                else:
                    formatted_numbers.append(f"<sign>{number.strip()}</sign>")
            return "Zeichen " + ', '.join(formatted_numbers)

        # Replace in the original text
        modified_text = pattern.sub(lambda m: replace_sign(m), text)
        return modified_text



    ############################# [2] PARAGRAPH AND SENTENCES  #########################################

    def reference_paragraphs_and_sentences(self,text, current_paragraph_id=None):
        # Regex patterns to capture various references
        # Absätzen 1 und 2  -> <par>current paragraph</par> Absätzen <sen>1</sen> und <sen>2</sen>
        # Absätzen 1 bis 1i beantragen ->  Absätzen <sen>1</sen> , <sen>1a</sen> , <sen>1b</sen> , <sen>1c</sen> , <sen>1d</sen> , <sen>1e</sen> , <sen>1f</sen> , <sen>1g</sen> , <sen>1h</sen> , <sen>1i</sen> beantragen
        
        
        paragraph_pattern = re.compile(r'§\s*(\d+\w*)')
        absatz_pattern = re.compile(r'\bAbsatz\s*(\d+\w*(?:\s*bis\s*\d+\w*)?)')  # Updated to capture ranges directly in regex
        
        #(§§ 36 bis 43  -> (<par>§ 36</par>, <par>§ 37</par>, <par>§ 38</par>, <par>§ 39</par>, <par>§ 40</par>, <par>§ 41</par>, <par>§ 42</par>, <par>§ 43</par> 
        range_paragraph_pattern = re.compile(r'§§\s*(\d+)\s*bis\s*(\d+)')
        

        

        # §§ 2 und 3 des Gesetzes -> <par>§ 2</par> und <par>§ 3</par> des Gesetzes
        multiple_paragraphs_pattern = re.compile(r'§§\s*(\d+)\s*und\s*(\d+)')
        

    
        # Function to replace single paragraph references with tags
        def paragraph_replacer(match):
            paragraph = match.group(1)
            return f'<par>§ {paragraph}</par>'

        def expand_range_paragraph(match):
            start, end = map(int, match.groups())
            paragraphs = ', '.join(f'§ {i}' for i in range(start, end + 1))
            return f'<par>{paragraphs}</par>'
        
    
        def expand_range_absatz(start, end):
            absatzes = ', '.join(f'<sen>{i}</sen>' for i in range(start, end + 1))
            return f'Absatz {absatzes}'

        def absatz_replacer(match):
            absatz_range = match.group(1)
            if 'bis' in absatz_range:
                # Split the range into start and end
                start, end = map(int, re.findall(r'\d+', absatz_range))
                return expand_range_absatz(start, end)
            pre_text = text[:match.start()]
            if re.search(r'<par>[^<]+</par>\s*$', pre_text):
                return f' Absatz <sen>{absatz_range}</sen>'
            return f'<par>{current_paragraph_id}</par> Absatz <sen>{absatz_range}</sen>' if current_paragraph_id else f' Absatz <sen>{absatz_range}</sen>'

        def multiple_paragraphs_replacer(match):
            first, second = match.groups()
            return f'<par>§ {first}</par> und <par>§ {second}</par>'

       

        # Apply all regex replacements
        text = re.sub(range_paragraph_pattern, expand_range_paragraph, text)  # Handle ranges first to avoid overlap issues
        text = re.sub(paragraph_pattern, paragraph_replacer, text)
        text = re.sub(absatz_pattern, absatz_replacer, text)
        text = re.sub(multiple_paragraphs_pattern, multiple_paragraphs_replacer, text)

        return text



    ############################# [3] ANLAGE AND ABSCHNITTE  #########################################
        
    def refrence_tables_and_sections(self,text):
        # Regex pattern to identify and format "Anlage" and "Abschnitt" references
        anlage_pattern = re.compile(r'\bAnlagen?\s+(\d+)(?:\s*bis\s*(\d+))?')
        abschnitt_pattern = re.compile(r'\bAbschnitt\s+(\d+)')

        # Function to replace "Anlage" references with tags
        def anlage_replacer(match):
            start, end = match.group(1), match.group(2)
            if end:
                return ', '.join(f'Anlage <tab>{i}</tab>' for i in range(int(start), int(end)+1))
            return f'Anlage <tab>{start}</tab>'

        # Function to replace "Abschnitt" references with tags
        def abschnitt_replacer(match):
            return f'Abschnitt <sec>{match.group(1)}</sec>'

        # Apply the regex patterns and their replacement functions to the text
        text = re.sub(anlage_pattern, anlage_replacer, text)
        text = re.sub(abschnitt_pattern, abschnitt_replacer, text)
        return text




