import requests
from bs4 import BeautifulSoup
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jiwer
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
import argparse

class ParserEval:
    def __init__(self,args,main_content, table_content):
        self.url = args.url
        self.save_scores = args.save_scores
        self.main_content = main_content
        self.table_content = table_content

    def get_html_content(self):
        response = requests.get(self.url)
        response.raise_for_status()  # Ensure the response status is OK
        html_content = response.content
        html_content = BeautifulSoup(html_content, 'html.parser')
        html_content = html_content.prettify()
        return  html_content
    
    def get_original_text_content(self):
        response = requests.get(self.url)
        response.raise_for_status()
        html_content = response.content
        html_content = BeautifulSoup(html_content, 'html.parser')
        text_content = html_content.get_text()
        # process the text content
        text_content = text_content.split('\n')
        text_content = [line.strip() for line in text_content]
        text_content = [line for line in text_content if line]
        text_content = '\n'.join(text_content)
        # remove links 
        text_content = text_content.split('\n')
        text_content = [line for line in text_content if 'http' not in line]
        text_content = '\n'.join(text_content)

        # concatenate all lines into a single paragraph
        text_content = text_content.split('\n')
        text_content = [line.strip() for line in text_content]
        text_content = [line for line in text_content if line]
        text_content = ' '.join(text_content)

        

        return text_content
    
    def convert_main_content_to_text(self):
        full_text = ''
        for category in self.main_content:
            full_text += category['category'] + '\n'
            for paragraph in category['paragraphs']:
                full_text += paragraph['paragraph'] + '\n'
                for sentence_dict in paragraph['sentences']:  # Assuming each section contains 'sentences' which is a list of dicts
                    for sen_id, sen_text in sentence_dict.items():
                        full_text += sen_text + '\n'
        
        # process the text content
        full_text = full_text.split('\n')
        full_text = [line.strip() for line in full_text]
        full_text = [line for line in full_text if line]
        full_text = '\n'.join(full_text)
        # remove links
        full_text = full_text.split('\n')
        full_text = [line for line in full_text if 'http' not in line]
        full_text = '\n'.join(full_text)

        # remove tags
        full_text = re.sub(r'<*>.*</*>', '', full_text)

        # concatenate all lines into a single paragraph
        full_text = full_text.split('\n')
        full_text = [line.strip() for line in full_text]
        full_text = [line for line in full_text if line]
        full_text = ' '.join(full_text)

        
        
        return full_text
    
    def convert_table_content_to_text(self):
        full_text = ''
        for table in self.table_content:
            full_text += table['table_name'] + '\n'
            for section in table['table_data']:
                full_text += section['Section'] + '\n'
                for row in section['Content']:
                    full_text += row['Sign_id']  +row['Sign_name'] + row['description'] + '\n'
                
        # process the text content
        full_text = full_text.split('\n')
        full_text = [line.strip() for line in full_text]
        full_text = [line for line in full_text if line]
        full_text = '\n'.join(full_text)
        # remove links
        full_text = full_text.split('\n')
        full_text = [line for line in full_text if 'http' not in line]
        full_text = '\n'.join(full_text)

        # remove tags 
        full_text = re.sub(r'<*>.*</*>', '', full_text)

        # concatenate all lines into a single paragraph
        full_text = full_text.split('\n')
        full_text = [line.strip() for line in full_text]
        full_text = [line for line in full_text if line]
        full_text = ' '.join(full_text)
    
        return full_text
    

    def calculate_tfidf_similarity(self,original_text, reconstructed_text):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([original_text, reconstructed_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])
        return similarity[0][0]
    
    def calculate_wer(self,original_text, reconstructed_text):
        wer = jiwer.wer(original_text, reconstructed_text)

        return wer

    def calculate_bleu_score(self, original_text, reconstructed_text):
        # Tokenize the texts
        reference = [[word for word in original_text.split()]]  # List of lists for corpus_bleu
        candidate = [word for word in reconstructed_text.split()]
        
        # Calculate sentence-level BLEU
        sentence_bleu_score = sentence_bleu(reference, candidate)
        
        
        return sentence_bleu_score
    
    def calculate_rouge_scores(self,original_text, reconstructed_text):
        # Initialize the Rouge object
        rouge = Rouge()
        
        # Compute the scores
        scores = rouge.get_scores(reconstructed_text, original_text)
        
        return scores
    
    def plot_scores(self, tfidf_score, wer_score, bleu_score):
        scores = [tfidf_score, wer_score, bleu_score]
        labels = ['TF-IDF', 'WER', 'BLEU']
        # colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # More distinct, visually appealing colors
        # suggest better colors combination , light blue, light green and light red
        colors = ['#ADD8E6', '#90EE90', '#FFA07A']
        



        # labels font size
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.figure(figsize=(10, 6))  # Slightly larger figure for better readability
        bars = plt.bar(labels, scores, color=colors)
        
        

        plt.xlabel('Similarity Metrics', fontsize=20, labelpad=10, fontweight='bold')
        plt.ylabel('Scores', fontsize=20, labelpad=10, fontweight='bold')
        plt.title('Original Legal Text vs Reconstructed Legal Text (StVZO)', fontsize=20, fontweight='bold')
        plt.ylim(0, 1)  # Ensuring all scores are visible and assuming normalization to [0,1]

        # Adding a grid for easier comparison
        plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        plt.gca().set_axisbelow(True)
        

    
        
        # add legend for the scores each color : score 
        # round the scores to 2 decimal places
        scores = [round(score, 2) for score in scores]
        plt.legend(bars, scores, loc='upper center', fontsize=20)
        

        

        # Improve layout to not cut off labels or titles
        plt.tight_layout()

        # Optionally, save the plot if a filename is provided
        if hasattr(self, 'save_scores'):
            plt.savefig(self.save_scores)

        



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract main content from a webpage')
    parser.add_argument('--url', type=str, default='https://www.gesetze-im-internet.de/stvo_2013/BJNR036710013.html', help='URL of the webpage')
    parser.add_argument('--base_url', type=str, default='https://www.gesetze-im-internet.de', help='Base URL of the webpage')
    parser.add_argument('--parsed_main', type=str , default="parsed_main_content_Straßenverkehrs_Ordnung.json", help='Directory to save the parsed main content')
    parser.add_argument('--parsed_table', type=str , default="parsed_table_content_Straßenverkehrs_Ordnung.json", help='Directory to save the parsed table content')
    parser.add_argument('--save_original', type=str , default="Straßenverkehrs_Ordnung_orignal.txt", help='Directory to save the original text content')
    parser.add_argument('--save_reconstructed', type=str , default="Straßenverkehrs_Ordnung_reconstructed.txt", help='Directory to save the reconstructed text content')
    parser.add_argument('--save_scores', type=str , default="Straßenverkehrs_Ordnung.png", help='Directory to save the scores plot')
    
    args = parser.parse_args()



    
    # read main content from json file
    try:
        with open(args.parsed_main, 'r') as f:
            main_content = json.load(f)
    except:
        main_content = []    
    # read table content from json file
    try :
        with open(args.parsed_table, 'r') as f:
            table_content = json.load(f)
    except:
        table_content = []
    
    parser = ParserEval(args, main_content, table_content)
    
    
    original_text = parser.get_original_text_content()
    # save the content to a file
    with open(args.save_original, 'w') as f:
        f.write(original_text)


    # convert main content to text
    main_text = parser.convert_main_content_to_text()
    # with open('main-content-highway-code.txt', 'w') as f:
    #     f.write(main_text)

    # convert table content to text
    table_text = parser.convert_table_content_to_text()
    # with open('table-content-highway-code.txt', 'w') as f:
    #     f.write(table_text)

    
    # combine both main and table content into a single file
    reconstructed_text = main_text + '\n' + table_text
    with open(args.save_reconstructed, 'w') as f:
        f.write(reconstructed_text)

    # calculate similarity scores
    tfidf_score = parser.calculate_tfidf_similarity(original_text, reconstructed_text)
    print('TF-IDF Similarity:', tfidf_score)
    wer_score = parser.calculate_wer(original_text, reconstructed_text)
    print('Word Error Rate:', wer_score)
    sentence_bleu  = parser.calculate_bleu_score(original_text, reconstructed_text)
    print('BLEU Score:', sentence_bleu)

    
    
    # plot the scores
    parser.plot_scores(tfidf_score, wer_score, sentence_bleu)

    
    # rouge_scores = parser.calculate_rouge_scores(original_text, reconstructed_text)
    # print('ROUGE Scores:', rouge_scores)
    
    # # plot rouge scores f1, precision and recall for rouge 1 and rouge 2 and rouge L
    # f1 = [rouge_scores[0]['rouge-1']['f'], rouge_scores[0]['rouge-2']['f'], rouge_scores[0]['rouge-l']['f']]
    # precision = [rouge_scores[0]['rouge-1']['p'], rouge_scores[0]['rouge-2']['p'], rouge_scores[0]['rouge-l']['p']]
    # recall = [rouge_scores[0]['rouge-1']['r'], rouge_scores[0]['rouge-2']['r'], rouge_scores[0]['rouge-l']['r']]
    # labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']

    # x = range(len(labels))
    # width = 0.2
    
    # plt.figure(figsize=(10, 6))
    # plt.bar(x, f1, width, label='F1', color='blue')
    # plt.bar([i + width for i in x], precision, width, label='Precision', color='green')
    # plt.bar([i + width*2 for i in x], recall, width, label='Recall', color='red')
    # plt.xlabel('ROUGE Scores')
    # plt.ylabel('Scores')
    # plt.title('Comparison of ROUGE Scores')
    # plt.xticks([i + width for i in x], labels)
    # plt.legend()
    # plt.savefig('rouge_scores_plot.png')
    

    
    

   
    


    



    
    
