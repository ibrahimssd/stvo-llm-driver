# write function to read json file  data/synthetic-data/stvo/qa.jsonl
# convert yes to correct  , no to incorrectin filed "label"
# {"question": "Is it mandatory to mark locked roads and routes using light sign devices?", "answer": "Only if the competent authority has ordered so, according to paragraph 3 of the regulation.", "label": "incorrect", "quality_score": 0.8, "paragraph_id": "\u00a7 45", "paragraph": "\u00a7 45Traffic Signs and Transport Facilities", "sentence": "Prior to the beginning of work affecting road traffic, the entrepreneurs \u2013 the construction entrepreneurs under the presentation of a road sign plan \u2013 shall request from the competent authority orders in accordance with paragraphs 1 to 3 on how to lock their workplaces and to mark whether and how to restrict, guide, including in the case of partial road lock, and to regulate is also whether and how they have to mark locked roads and routes. you have to follow these orders and use light sign devices.", "is_yes_no_question": true, "language": "en", "generation_attempt": 1, "sentence_length": 504}
# {"question": "Can search lights be used to illuminate pedestrian crossings?", "answer": "No, pursuant to section 17 of the Lighting Regulations, search lights may only be used to light trails.", "label": "incorrect", "quality_score": 0.8, "paragraph_id": "\u00a7 17", "paragraph": "\u00a7 17Lighting", "sentence": "(6) Search panels may only be used short and not to light the trail.", "is_yes_no_question": true, "language": "en", "generation_attempt": 1, "sentence_length": 68}

import json
import random
def replace_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    for item in data:
        if item['label'] == 'yes':
            item['label'] = 'correct'
        elif item['label'] == 'no':
            item['label'] = 'incorrect'

    with open(output_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
# ensure the test size is 2084 samples
def split_train_test_by_paragraph_id(input_file, train_file, test_file, test_size=0.16):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Group data by paragraph_id
    paragraph_groups = {}
    for item in data:
        pid = item['paragraph_id']
        if pid not in paragraph_groups:
            paragraph_groups[pid] = []
        paragraph_groups[pid].append(item)

    # Split paragraph groups into train and test
    paragraph_ids = list(paragraph_groups.keys())
    random.shuffle(paragraph_ids)
    split_index = int(len(paragraph_ids) * (1 - test_size))
    train_paragraphs = paragraph_ids[:split_index]
    test_paragraphs = paragraph_ids[split_index:]

    # Write train and test files
    with open(train_file, 'w') as train_f, open(test_file, 'w') as test_f:
        for pid in train_paragraphs:
            for item in paragraph_groups[pid]:
                json.dump(item, train_f)
                train_f.write('\n')
        for pid in test_paragraphs:
            for item in paragraph_groups[pid]:
                json.dump(item, test_f)
                test_f.write('\n')

if __name__ == "__main__":    
    input_file = '../synthetic-data/stvo/qa.jsonl'
    output_file = '../synthetic-data/stvo/qa.jsonl'
    replace_data(input_file, output_file)

    # split train and test data by paragraph id 
    train_file = '../synthetic-data/stvo/qa_train.jsonl'
    test_file = '../synthetic-data/stvo/qa_test.jsonl'
    # split_train_test_by_paragraph_id(output_file, train_file, test_file)
    # print some stats about the new files 
    with open(train_file, 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f]

    print(f"Train data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    # paragraphs in test data should not be in train data
    train_paragraphs = set(item['paragraph_id'] for item in train_data)
    test_paragraphs = set(item['paragraph_id'] for item in test_data)
    assert len(train_paragraphs.intersection(test_paragraphs)) == 0, "Paragraphs overlap between train and test!"
    print("No paragraph overlap between train and test data. Data preprocessing complete.")

    # number of pargaraphs in train and test data
    print(f"Unique paragraphs in train data: {len(train_paragraphs)}")
    print(f"Unique paragraphs in test data: {len(test_paragraphs)}")

    # duplicate both questions and answers in train and test data (has same combinaition between question and answer)
    train_qa_pairs = set((item['question'], item['answer']) for item in train_data)
    test_qa_pairs = set((item['question'], item['answer']) for item in test_data)
    print(f"Unique question-answer pairs in train data: {len(train_qa_pairs)}")
    print(f"Unique question-answer pairs in test data: {len(test_qa_pairs)}")

    # correct vs incorrect distribution in train and test data
    train_label_distribution = {'correct': 0, 'incorrect': 0}
    for item in train_data:
        train_label_distribution[item['label']] += 1
    test_label_distribution = {'correct': 0, 'incorrect': 0}
    for item in test_data:
        test_label_distribution[item['label']] += 1
    print(f"Train label distribution: {train_label_distribution}")
    print(f"Test label distribution: {test_label_distribution}")
    # percentage of correct vs incorrect in train and test data
    train_correct_percentage = train_label_distribution['correct'] / len(train_data) * 100
    train_incorrect_percentage = train_label_distribution['incorrect'] / len(train_data) * 100
    test_correct_percentage = test_label_distribution['correct'] / len(test_data) * 100
    test_incorrect_percentage = test_label_distribution['incorrect'] / len(test_data) * 100
    print(f"Train correct percentage: {train_correct_percentage:.2f}%")
    print(f"Train incorrect percentage: {train_incorrect_percentage:.2f}%")
    print(f"Test correct percentage: {test_correct_percentage:.2f}%")
    print(f"Test incorrect percentage: {test_incorrect_percentage:.2f}%")
    