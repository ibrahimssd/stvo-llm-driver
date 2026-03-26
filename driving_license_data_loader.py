import json
import os
from typing import List, Dict


class DrivingLicenseDataHandler:
    """
    Handles reading and processing driving license question data from JSON files.
    """

    def read_german_questions_answers(self, file_path: str, skip_category: str = 'direct_answer', skip_images: bool = True) -> List[Dict]:
        """
        Reads German driving license questions and answers from a JSON file.

        Args:
            file_path: Path to the JSON file.
            skip_category: Category of questions to skip.
            skip_images: Whether to skip questions with images.

        Returns:
            A list of dictionaries, where each dictionary represents a question and its details.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON from {file_path}: {e}")
            return []

        question_answers_list = []
        for item in data:
            if (skip_images and 'image' in item) or (item.get('category') == skip_category):
                continue
            image_id = item.get('image', None) if not skip_images and 'image' in item else None
            id = item.get('id', 'No ID provided')
            question = item.get('question', 'No question provided')
            options = item.get('options', [])
            answers = item.get('answers', [])
            category = item.get('category', 'No category provided')

            question_answers_list.append({
                'id': id,
                'category': category,
                'question': question,
                'options': options,
                'expected_answers': answers,
                'image_id': image_id,
            })

        return question_answers_list

    def read_irish_questions_answers(self, file_path: str, skip_images: bool = True) -> List[Dict]:
        """
        Reads Irish driving license questions and answers from a JSON file.

        Args:
            file_path: Path to the JSON file.
            skip_images: Whether to skip questions with images.

        Returns:
            A list of dictionaries, where each dictionary represents a question and its details.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON from {file_path}: {e}")
            return []

        question_answers_list = []
        for item in data:
            if skip_images and 'image' in item:
                continue
            question = item.get('question', 'No question provided')
            heading = item.get('heading', 'No heading provided')
            explanation = item.get('explanation', 'No explanation provided')
            options = item.get('options', [])
            answers = item.get('answers', [])
            id = item.get('id', 'No ID provided')
            image_id = item.get('image') if not skip_images and 'image' in item else None

            question_answers_list.append({
                'heading': heading,
                'question': question,
                'options': options,
                'expected_answers': answers,
                'explanation': explanation,
                'id': id,
                'image_id': image_id
            })

        return question_answers_list

    def read_austrian_questions_answers(self, directory_path: str, language: str = 'de', skip_images: bool = True) -> List[Dict]:
        """
        Reads Austrian driving license questions and answers from JSON files in a directory.

        Args:
            directory_path: Path to the directory containing the JSON files.
            language: Language of the questions and answers ('de' or 'en').
            skip_images: Whether to skip questions with images.

        Returns:
            A list of dictionaries, where each dictionary represents a question and its details.
        """
        question_answers_list = []
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

        for filename in json_files:
            file_path = os.path.join(directory_path, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    item = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON from file: {file_path}, error: {e}")
                continue
            except IOError as e:
                print(f"IO error when opening file: {file_path}, error: {e}")
                continue

            if skip_images and 'BildName' in item:
                continue

        
            image_id = item.get('BildName') if not skip_images and 'BildName' in item else None
            question_id = item.get('FRAGEID')
            theme_id = item.get('THEMAID')
            difficulty_id = item.get('SCHWIERIGID')
            question_text = item.get('FrageText', {}).get(language, "")
            answers = item.get('AntwortText', {}).get(language, [])
            correct_answers = item.get('AntwortRichtig', [])

            if question_text and answers:
                paired_answers = [{'text': ans['Text'], 'correct': ca} for ans, ca in zip(answers, correct_answers)]
                expected_answers = [pa['text'] for pa in paired_answers if pa['correct']]

            question_answers_list.append({
                'question_id': question_id,
                'theme_id': theme_id,
                'difficulty_id': difficulty_id,
                'question': question_text,
                'options': [ans['Text'] for ans in answers],
                'correct_answers': correct_answers,
                'paired_answers': paired_answers,
                'expected_answers': expected_answers,
                'image_id': image_id,
            })

        return question_answers_list



