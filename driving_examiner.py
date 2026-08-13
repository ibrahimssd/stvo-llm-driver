import json
import logging
import os
import torch
import re
from transformers import AutoTokenizer
import sympy as sp
from typing import List, Dict, Union, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef , balanced_accuracy_score
from multi_task_classes_STvO_mlm_clm_clu_cls import  TokenizerManager, UnifiedMultiTaskModel

# from rag import RAG
from PIL import Image
import logging


# Set up logging
# logging.basicConfig(level=logging.INFO)


class DrivingLicenseExaminer:
    """
    Evaluates Large Language Models (LLMs) on driving license theory test questions.
    Supports multiple-choice and direct answer question types.
    """

    def __init__(self, args):
        """
        Initializes the DrivingLicenseExaminer with model, tokenizer, and configurations.

        Args:
            args: Configuration arguments containing model paths, decoding strategies, etc.
        """
        self.args = args
        self.access_token = os.environ.get("HF_TOKEN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls_num_labels = self.args.num_cls_labels 
        self.model, self.tokenizer = self._load_models()
        self.model.eval()
        # self._setup_tokenizer()
        self._setup_prompt_templates()
        self._setup_context_variables()
        # self.rag = RAG(args)
        

        # Centralize generation arguments
        self.generation_args = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": True,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "temperature": self.args.temperature,
            "repetition_penalty": self.args.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "num_return_sequences": 1,
        }
        

    def _load_models(self) -> Tuple[UnifiedMultiTaskModel, AutoTokenizer]:
        """Loads the model and tokenizer from pretrained weights."""
        try:
            # --- Tokenizer ---
            tokenizer_manager = TokenizerManager(access_token=os.environ.get("HF_TOKEN"))
            tokenizer = tokenizer_manager.initialize_tokenizer(self.args.baseline_model,self.args.cache_dir,
                                                                self.args.modeling_type)
            
            model_name = self.args.baseline_model.split("/")[-1] # Extract model name from path
            with open(self.args.paragraph_embeddings_file, 'r', encoding='utf-8') as f:
                self.paragraph_embeddings = json.load(f)
            # embedding_dim = next(iter(paragraph_embeddings.values())).shape[0]
            embedding_dim = len(next(iter(self.paragraph_embeddings.values()))) if self.paragraph_embeddings else 0

            
            # model = MultiTaskModelClustering(
            #     base_model_name=self.args.baseline_model,
            #     access_token=os.environ.get("HF_TOKEN"),
            #     embedding_dim=embedding_dim,
            #     tokenizer_len=len(tokenizer),
            #     mask_token_id=tokenizer.mask_token_id,
            #     num_cls_labels=self.cls_num_labels,
            #     HF_CACHE_DIR=self.args.cache_dir,
            #     modeling_type=self.args.modeling_type,
            #     max_segments=5,
            #     clu_max_segment_length=self.args.max_seq_length_clu
            # )

            model= UnifiedMultiTaskModel(
                base_model_name=self.args.baseline_model,
                access_token=tokenizer_manager.access_token,
                embedding_dim=embedding_dim,
                paragraph_embeddings=self.paragraph_embeddings,
                tokenizer_len=len(tokenizer),
                mask_token_id=tokenizer.mask_token_id, # Use tokenizer's mask token ID
                num_cls_labels=2,
                max_segments=5,
                clu_max_segment_length=self.args.max_seq_length_clu,
                nsp_max_segment_length=self.args.max_seq_length_nsp,
                cls_max_segment_length=self.args.max_seq_length_cls,
                hf_cache_dir=self.args.cache_dir,
                modeling_type=self.args.modeling_type,
            )
            
            
                
            # --- CONDITIONAL CHECKPOINT LOADING ---
            if not getattr(self.args, 'eval_base', False): # Use getattr for safety, default False
                # This branch is for evaluating a fine-tuned model from a checkpoint
                if not os.path.exists(self.args.model_checkpoint_dir):
                    logging.error(f"Checkpoint file not found at {self.args.model_checkpoint_dir}. Cannot load fine-tuned model.")
                    raise FileNotFoundError(f"Checkpoint file not found: {self.args.model_checkpoint_dir}")

                logging.info(f"Loading model state dictionary from {self.args.model_checkpoint_dir}...")
                # Load the state dict, mapping it to the correct device
                model.load_state_dict(torch.load(f"{self.args.model_checkpoint_dir}/{model_name}_{self.args.model_checkpoint}"))
                logging.info("Model state dictionary loaded successfully.")
            else:
                # This branch is for evaluating the base model (without loading a fine-tuned checkpoint)
                logging.info("Evaluating base model without loading a fine-tuned checkpoint.")

            model.to(self.device)
            return model, tokenizer
                
        except Exception as e:
            logging.error(f"Failed to load model or tokenizer: {e}")
            raise
    

    def _setup_prompt_templates(self) -> None:
        """Defines prompt templates for different question types."""
        self.MCQs_SYSTEM_PROMPT = (
            f"You are an expert evaluator. Your task is to determine the correctness of a given answer to a question for the {self.args.data_name} driving theory test. "
            "You MUST respond with a single word: 'yes' or 'no'."
            "Do not provide any additional text, explanations, or punctuation."
        )
        #  f"You are an experienced driver helping to answer {self.args.data_name} driving theory test questions. "
        #     "For each option provided, determine if it is correct or incorrect. Respond with 'yes' if the option is correct, "
        #     "and 'no' if the option is incorrect, without any additional explanation."
        self.DIRECT_SYSTEM_PROMPT = (
            f"As an experienced driver, your task is to provide answers to {self.args.data_name} driving theory test questions. "
            "Please respond with the exact numerical value or mathematical formula or sign names required by the question, "
            "without including any extraneous information or explanation."
        )

    def _setup_context_variables(self) -> None:
        """Defines context variables for symbolic algebraic comparisons."""
        self.context_variables = {
            'speed': sp.symbols('speed'),
            'reaction_path': sp.symbols('reaction_path'),
            'braking_distance': sp.symbols('braking_distance'),
            'm': sp.symbols('m')
        }

    def evaluate_batch_model(self, questions_answers: List[Dict], inference_batch_size: int) -> Dict[str, Union[float, int]]:
        """Evaluates the model on a batch of questions."""
        all_predictions, all_references = [], []
        for i in range(0, len(questions_answers), inference_batch_size):
            batch = questions_answers[i:i + inference_batch_size]
            if self.args.skip_category == 'direct_answer' and self.args.skip_images:
                logging.info("Processing batch with textual MCQs...")
                predictions, references = self._process_batch_with_textual_mcqs(batch)
            elif self.args.skip_category == 'direct_answer' and not self.args.skip_images:
                logging.info("Processing batch with images...")
                predictions, references = self._process_batch_with_images(batch)
            elif self.args.skip_category == 'multiple_choice':
                logging.info("Processing batch with direct answer questions...")
                predictions, references = self._process_batch_with_textual_direct_questions(batch)
            else:
                logging.error("Invalid skip category specified. Please choose skipping category 'direct_answer' or 'multiple_choice'.")
                continue
            all_predictions.extend(predictions)
            all_references.extend(references)
        return self._calculate_and_log_metrics(all_predictions, all_references, len(questions_answers))

    def _process_batch_with_textual_mcqs(self, batch: List[Dict]) -> Tuple[List[str], List[str]]: 
        """Processes a batch of multiple-choice questions."""
        batch_prompts = [self._generate_text_prompt(qa['question'], option, self.args.prompt_version) for qa in batch for option in qa['options']]
        reference_responses = ['yes' if option in qa['expected_answers'] else 'no' for qa in batch for option in qa['options']]
        # batch_responses = [self._generate_response(user_prompt, self.MCQs_SYSTEM_PROMPT) for user_prompt in batch_prompts]
        batch_responses = [self._predict_cls_yes_no(qa['question'], option) for qa in batch for option in qa['options']]
        # batch_responses = [self._generate_yes_no_response_similarity(qa['question'], option) for qa in batch for option in qa['options']]
        predictions = [self._evaluate_response(response) for response in batch_responses]
        self._log_batch_results(batch_prompts, reference_responses, batch_responses, predictions, self.MCQs_SYSTEM_PROMPT)
        return predictions, reference_responses
  
    def _process_batch_with_textual_direct_questions(self, batch: List[Dict]) -> Tuple[List[str], List[str]]:
        """Processes a batch of direct answer questions."""
        batch_prompts = [self._generate_direct_question_prompt(qa['question'], self.args.prompt_version) for qa in batch]
        reference_responses= ['yes' if qa['expected_answers'] else 'no' for qa in batch]
        reference_answers = [qa['expected_answers'] for qa in batch]
        batch_responses = [self._generate_response(user_prompt, self.DIRECT_SYSTEM_PROMPT) for user_prompt in batch_prompts]
        predictions = [self._evaluate_response_with_direct_questions(response, ref_answer) for response, ref_answer in zip(batch_responses, reference_answers)]
        self._log_direct_batch_results(batch_prompts, reference_answers, reference_responses, batch_responses, predictions, self.DIRECT_SYSTEM_PROMPT)
        return predictions, reference_responses
    

    ## Prompt Generation Functions##
    def _generate_text_prompt(self, question: str, answer: str, version: int = 1) -> str:
        """Generates text prompts for multiple-choice questions."""
        prompts = {
            1: f"Given the question: '{question}', determine if the answer '{answer}' is correct.Please Respond only with 'yes' if the answer is correct , or 'no' if the answer is not correct; without any additional explanation.",
            2: f"Question: '{question}' Is the answer '{answer}' correct? Respond with 'yes' or 'no' only.",
            3: f"Analyze the following question: '{question}' with the proposed answer '{answer}'. Is this answer correct? Please reply with 'yes' or 'no' only.",
            4: f"Regarding the question: '{question}', evaluate the correctness of the answer '{answer}'. Kindly limit your response to 'yes' for correct answer and 'no' for incorrect answer.",
            5: f"Check this statement: '{question}' Answer: '{answer}'. Correct? Reply 'yes' or 'no'.",
            6: f"Given the question '{question}', can you confirm if the corresponding answer '{answer}' is accurate? Please respond simply with 'yes' if it is accurate, or 'no' if it is not.",
            7: f"question: '{question}' Is the answer '{answer}' true?. Respond only with 'yes' if true or 'no' if false.",
            8: f"""Question: {question}
                   Answer: {answer}
                   Is this answer correct? Respond with 'yes' or 'no' only.""",
            10: f"""You are an expert evaluator. Your task is to determine the correctness of a given answer to a question for the {self.args.data_name} driving theory test.
                ---
                Question: "In which instances do you have to approach a pedestrian crossing with particular care?"
                Answer: "If the view of the pedestrian crossing is restricted"
                Response: yes

                Question: "How can you tell while driving that an indicator lamp is not working?"
                Answer: "You cannot detect a defective indicator lamp while driving"
                Response: no

                Question: "When may a tram be overtaken on the left?"
                Answer: "When the rails run too far to the right"
                Response: yes

                Question: "{question}"
                Answer: "{answer}"
                Response:""" , # Note: The last line is left open for the model to fill in.
        }

        # _retrieve_relevant_paragraphs
        # prompt = (
        #         f"Based on the following legal text, answer the with 'yes' or 'no' only. "
        #         f"Legal Text: {self._retrieve_relevant_paragraphs(question, answer, top_k=3)}\n"
        #         f"{prompts.get(version, prompts[1])}"  # Default to version 1 if not found  
        #     )
        return prompts.get(version, prompts[1])  # Default to version 1 if not found


    def _generate_direct_question_prompt(self, question: str, version: int = 1) -> str:
        """Generates prompts for direct answer questions."""
        if version == 1:
            return f"For the question: '{question}', please provide a numeric answer or a mathematical expression or sign names, as required."
        return ""
    
    
    
    # CLS task prediction function
    def _predict_cls_yes_no(self, question: str, answer: str) -> str:
        """
        Predicts 'yes' or 'no' using the fine-tuned MultiTaskModelClustering
        with the 'cls' task head.
        """
        # 1. Set the model to evaluation mode.
        # This disables dropout and other layers that behave differently during training.
        self.model.eval()

        # 2. Combine the question and answer into a single input string.
        input_text = f"Question: {question} [SEP] Answer: {answer}"

        # 3. Tokenize the input string and move tensors to the correct device.
        # The tokenizer must be the same one used during training.
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.args.max_seq_length_cls,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        # 4. Run the forward pass with no gradient calculation.
        # We pass 'cls' as the task_type and do NOT pass any labels.
        with torch.no_grad():
            # The forward method returns a tuple of (clm_logits, clu_logits, cls_logits).
            # We only need the third element, which contains the classification logits.
        
            results = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_type=self.args.task_type  # Use 'cls' for classification
            )
            cls_logits = results['cls_logits']
                

        # 5. Get the predicted class ID by finding the index with the highest logit.
        predicted_class_id = torch.argmax(cls_logits, dim=1).item()

        # 6. Map the predicted ID back to the string label.
        # This mapping should be consistent with the one used in the dataset loading.
        id_to_label = {0: "no", 1: "yes"}
        
        return id_to_label[predicted_class_id]

        

    def _retrieve_relevant_paragraphs(self, question: str, answer: str, top_k: int) -> List[str]:
        """Retrieves relevant paragraphs based on the question and answer."""
        # This is a placeholder function. Implement retrieval logic as needed.
        # return ["Relevant paragraph 1", "Relevant paragraph 2"]

        with torch.no_grad():
            # --- NEW: Hierarchical Tokenization Logic ---
            # Instead of tokenizing the full text, we prepare it for the hierarchical model.
        
            input_text = question + " " + answer

                
            # Heuristic: split by double newline to get paragraphs.
            paragraphs = input_text.split('. ')
            paragraphs = [p for p in paragraphs if p.strip()]


            # Truncate or pad paragraphs to fit max_segments
            if len(paragraphs) > self.args.max_num_segments:
                paragraphs = paragraphs[:self.args.max_num_segments]

            # Tokenize each paragraph individually
            tokenized_paragraphs = self.tokenizer(
                paragraphs,
                padding='max_length',
                truncation=True,
                max_length=self.args.max_seq_length_clu,
                return_tensors='pt'
            )
                
            # Get the tensors and ensure they are of the correct size
            # If a document has fewer than max_segments, we pad with empty tensors
            input_ids = tokenized_paragraphs['input_ids']
            attention_mask = tokenized_paragraphs['attention_mask']

            
            # --- Get sentence embeddings from the hierarchical model ---
            # The model's forward pass now knows how to handle the 3D tensor
            results = self.model(
                input_ids=input_ids.unsqueeze(0).to(self.device),  # Add batch dimension
                attention_mask=attention_mask.unsqueeze(0).to(self.device),  # Add batch dimension
                task_type='clu'
            )
            input_text_embeddings = results['clu_logits']

            if input_text_embeddings is None:
                raise RuntimeError("Model returned None for clustering logits. Check task_type or forward method logic.")
            
            # Compare the question embedding to all paragraph embeddings
            paragraph_ids = list(self.paragraph_embeddings.keys())
            paragraph_tensors = torch.tensor(list(self.paragraph_embeddings.values()), dtype=torch.float).to(self.device)
            
            # Use cosine similarity to find the most relevant paragraphs
            similarities = torch.nn.functional.cosine_similarity(input_text_embeddings, paragraph_tensors, dim=1)
            top_k_indices = torch.topk(similarities, k=top_k).indices

            relevant_paragraphs = [paragraph_ids[i] for i in top_k_indices]
            print(f"Retrieved Paragraphs: {relevant_paragraphs}")
            
            return relevant_paragraphs


    ## Response Generation Functions ##
    def _generate_yes_no_response_similarity(self, question: str, answer: str) -> str:
        """Generates a yes/no response for a given question and answer."""
        # 2. Prepare the paragraph embeddings (the "knowledge base")
        # Ensure paragraph_ids and paragraph_tensors are in the same order
        paragraph_ids = list(self.paragraph_embeddings.keys())
        paragraph_tensors = torch.tensor(list(self.paragraph_embeddings.values()), dtype=torch.float).to(self.device)

        # 3. Disable gradients for inference
        with torch.no_grad():
            # 4. Tokenize the new sentences
            question_inputs = self.tokenizer(
                [question],
                padding=True,
                truncation=True,
                max_length=self.args.max_seq_length_clu,
                return_tensors="pt"
            ).to(self.device)
            # 5. Get the question embeddings
            _, question_embeddings = self.model(
                input_ids=question_inputs["input_ids"],
                attention_mask=question_inputs["attention_mask"],
                task_type=self.args.task_type
            )

            # get answer embeddings
            answer_inputs = self.tokenizer(
                [answer],
                padding=True,
                truncation=True,
                max_length=self.args.max_seq_length_clu,
                return_tensors="pt"
            ).to(self.device)
            _, answer_embeddings = self.model(
                input_ids=answer_inputs["input_ids"],
                attention_mask=answer_inputs["attention_mask"],
                task_type=self.args.task_type
            )
            # 6. Calculate cosine similarity between question and answer embeddings 
            question_answer_similarity = torch.cosine_similarity(question_embeddings, answer_embeddings, dim=-1)
            # logging.info(f"Question-Answer Similarity: {question_answer_similarity.item()}")

            # return yes or no based on the similarity score
            if question_answer_similarity.item() > self.args.similarity_threshold:
                return 'yes'
            else:
                return 'no'
            



    def _generate_response(self, user_prompt: str, system_prompt: str) -> str:
        """Generates a response from the model."""
        
        # 1. Combine prompts into a single input string.
        # The prompt should be self-contained for the model to generate a response.
        model_prompt = str(system_prompt) + " " + str(user_prompt)
        
        inputs = self.tokenizer(model_prompt, return_tensors="pt").to(self.device)

        # 2. Define generation arguments.
        # We use do_sample=False for a deterministic, factual output.
        # This also means other sampling parameters are ignored and can be omitted.
        # You can easily change this to do_sample=True for a creative task.
        generation_args = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": False,
            "repetition_penalty": self.args.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "num_return_sequences": 1,
        }

        # 3. Generate the response using the language model.
        with torch.no_grad():
            generated_output = self.model.lm_model.generate(
                **inputs,
                **generation_args
            )
        
        # 4. Decode the output and remove the original prompt.
        # This approach is robust and ensures no extra text is included.
        generated_text = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)
        response = generated_text.replace(model_prompt, '').strip()
        
        # Optional: Log a warning if no response was generated.
        if not response:
            logging.warning("Model generated an empty response.")

        return response

    
        
    ########## Response Evaluation Functions ##########
    def _evaluate_response(self, generated_answer: str) -> str:
        """Evaluates a multiple-choice response by parsing the first word."""
        positive_words = {'yes', 'correct', 'true'}
        negative_words = {'no', 'incorrect', 'wrong', 'false'}
        generated_answer_lower = generated_answer.strip().lower()

        # Find all 'yes' or 'no' words
        pattern = r'\b(' + '|'.join(map(re.escape, positive_words)) + r'|' + '|'.join(map(re.escape, negative_words)) + r')\b'
        matches = re.findall(pattern, generated_answer_lower)

        if matches:
            first_word = matches[0]
            if first_word in positive_words:
                return 'yes'
            elif first_word in negative_words:
                return 'no'
        
        # If no clear 'yes' or 'no' is found, assume 'no' or 'invalid'
        return 'no'

    # def _evaluate_response(self, generated_answer: str) -> str:
    #     """Evaluates a multiple-choice response."""
    #     # positive_words = {'yes', 'correct', 'true'}
    #     positive_words = {'yes'}
    #     # negative_words = {'no', 'incorrect', 'wrong', 'not correct', 'false', 'not true'}
    #     negative_words = {'no'}
    #     generated_answer_lower = generated_answer.strip().lower()
    #     affirmative_pattern = r'\b(' + '|'.join(map(re.escape, positive_words)) + r')\b'
    #     contains_affirmative = bool(re.search(affirmative_pattern, generated_answer_lower))
    #     negative_pattern = r'\b(' + '|'.join(map(re.escape, negative_words)) + r')\b'
    #     contains_negative = bool(re.search(negative_pattern, generated_answer_lower))

    #     if contains_affirmative and not contains_negative:
    #         return 'yes'
    #     # if contains_negative and not contains_affirmative:
    #     #     return 'no'

    #     return 'no'

    ################  To be modified  ################
    def _evaluate_response_with_direct_questions(self, generated_answer: str, reference_answer: Union[str, List[str]]) -> str:
        """
        Evaluates the generated answer against the reference answer by first trying to interpret
        and compare them as algebraic expressions, numeric values, or through text comparison.
        
        :param generated_answer: The answer generated by the system.
        :param reference_answer: The correct answer(s) provided in the reference.
        :return: "yes" if the answers match either algebraically, numerically, or through text comparison, "no" otherwise.
        """
        def extract_numbers(text):
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)
            # Extracting floating point numbers and integers from the text
            return set(map(float, re.findall(r'\b\d+(?:\.\d+)?\b', text)))

        try:
            # Convert strings to symbolic expressions, incorporating context variables
            expr1 = sp.sympify(generated_answer, locals=self.context_variables)
            if isinstance(reference_answer, list):
                # Evaluate each reference answer as an expression
                results = [sp.simplify(expr1 - sp.sympify(ans, locals=self.context_variables)).is_zero for ans in reference_answer]
                if any(results):
                    return "yes"
            else:
                expr2 = sp.sympify(reference_answer, locals=self.context_variables)
                if sp.simplify(expr1 - expr2).is_zero:
                    return "yes"
        except Exception as e:
            print(f"Error in symbolic comparison: {str(e)}")

        # If algebraic comparison fails, check for numerical equality
        generated_numbers = extract_numbers(generated_answer)
        if isinstance(reference_answer, list):
            for ans in reference_answer:
                reference_numbers = extract_numbers(ans)
                if generated_numbers == reference_numbers:
                    return "yes"
        else:
            reference_numbers = extract_numbers(reference_answer)
            if generated_numbers == reference_numbers:
                return "yes"

        return "no"

    ############## Metrics Calculation and Logging Functions ##############
    def _calculate_and_log_metrics(self, predictions: List[str], references: List[str], num_questions: int) -> Dict[str, Union[float, int]]:
        """Calculates and logs evaluation metrics."""
        accuracy, balanced_accuracy, precision, recall, f1_macro, f1_micro, mcc = self.calculate_metrics(predictions, references)
        statistics = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "MCC": mcc,
            "number_of_questions": num_questions,
            "number_of_subquestions": len(predictions),
            "number_true_yes": references.count('yes'),
            "number_true_no": references.count('no'),
            "number_pred_yes": predictions.count('yes'),
            "number_pred_no": predictions.count('no'),
            "number_pred_invalid": predictions.count('invalid')
        }
        logging.info(f"Evaluation Metrics: {statistics}")
        return statistics
    
    def calculate_metrics(self, predictions: List[str], references: List[str]) -> Tuple[float, float, float, float, float]:
        """Calculates evaluation metrics for a list of predictions and references."""
        accuracy = accuracy_score(references, predictions)
        balanced_accuracy = balanced_accuracy_score(references, predictions)
        precision = precision_score(references, predictions, average='macro')
        recall = recall_score(references, predictions, average='macro')
        f1_macro = f1_score(references, predictions, average='macro')
        f1_micro = f1_score(references, predictions, average='micro')
        MCC = matthews_corrcoef(references, predictions)
        return accuracy, balanced_accuracy, precision, recall, f1_macro, f1_micro, MCC

    def _log_batch_results(self, prompts: List[str], references: List[str], responses: List[str], predictions: List[str], system_prompt: str) -> None:
        """Logs results for a batch of multiple-choice questions."""
        logging.info("SYSTEM PROMPT: " + system_prompt)
        for i, (prompt, ref, gen, pred) in enumerate(zip(prompts, references, responses, predictions)):
            logging.info(f"Prompt {i + 1}: {prompt}")
            logging.info(f"Expected Answer: {ref}")
            logging.info(f"Generated Output: {gen}")
            logging.info(f"Prediction: {pred}")
            logging.info(f"Is Correct: {pred.strip().lower() == ref.strip().lower()}")
            logging.info("-" * 50)

    def _log_direct_batch_results(self, prompts: List[str], reference_answers: List[Union[str, List[str]]], reference_predictions: List[str], responses: List[str], predictions: List[str], system_prompt: str) -> None:
        """Logs results for a batch of direct answer questions."""
        logging.info("SYSTEM PROMPT: " + system_prompt)
        for i, (prompt, ref_answ, ref_pred, gen, pred) in enumerate(zip(prompts, reference_answers, reference_predictions, responses, predictions)):
            logging.info(f"Prompt {i + 1}: {prompt}")
            logging.info(f"reference  Answer: {ref_answ}")
            logging.info(f"reference prediction: {ref_pred}")
            logging.info(f"Generated Output: {gen}")
            logging.info(f"Generated Prediction: {pred}")
            logging.info(f"Is Correct: {pred.strip().lower() == ref_pred.strip().lower()}")
            logging.info("-" * 50)

    