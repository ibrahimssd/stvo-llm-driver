from driving_license_data_loader import DrivingLicenseDataHandler
import logging
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json

logger = logging.getLogger(__name__)


def separate_correct_incorrect(questions_answers):
    positive = []
    negative = []
    for qa in questions_answers:
        question = qa['question']
        for paired in qa.get('paired_answers', []):
            entry = {
                'question': question,
                'answer': paired['text'],
                'label': paired['correct']
            }
            if paired['correct']:
                positive.append(entry)
            else:
                negative.append(entry)

    return positive, negative

# --- Embedding and Similarity Functions ---
def get_cls_embedding(text, model, tokenizer):
    """
    Generates a sentence embedding using the [CLS] token output of a BERT-based model.
    
    Args:
        text (str): The input sentence or text.
        model (PreTrainedModel): The loaded Legal-BERT model.
        tokenizer (PreTrainedTokenizer): The loaded Legal-BERT tokenizer.
        
    Returns:
        np.ndarray: A 1D NumPy array representing the [CLS] embedding.
    """
    # Tokenize the input and convert it to PyTorch tensors
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(**inputs)
        
        # The first element of the last hidden state is the [CLS] token embedding
        # Shape: (batch_size, sequence_length, hidden_size) -> We take the first token (index 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
    # Convert the PyTorch tensor to a NumPy array
    return cls_embedding.squeeze().cpu().numpy()

def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two 1D NumPy vectors.
    
    Args:
        vec_a (np.ndarray): First embedding vector.
        vec_b (np.ndarray): Second embedding vector.
        
    Returns:
        float: The cosine similarity score (-1.0 to 1.0).
    """
    # Use NumPy's dot product and linear algebra norm function
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

# --- Main Evaluation Function ---

def evaluate_similarity(positive_samples, negative_samples, model, tokenizer):
    """
    Calculates and compares similarity scores for positive (similar) and 
    negative (dissimilar) legal sentence pairs.

    Args:
        positive_samples (list of tuples): Pairs of similar legal texts: [(text1, text2), ...].
        negative_samples (list of tuples): Pairs of dissimilar legal texts: [(text1, text2), ...].
        model (PreTrainedModel): The loaded Legal-BERT model.
        tokenizer (PreTrainedTokenizer): The loaded Legal-BERT tokenizer.

    Returns:
        tuple: (positive_scores, negative_scores, avg_positive, avg_negative, distance)
    """
    
    positive_scores = []
    negative_scores = []

    # 1. Process Positive Samples
    print(f"Processing {len(positive_samples)} positive samples...")
    for text_a, text_b in positive_samples:
        # Generate [CLS] embeddings for both texts
        emb_a = get_cls_embedding(text_a, model, tokenizer)
        emb_b = get_cls_embedding(text_b, model, tokenizer)
        
        # Calculate similarity and store
        score = calculate_cosine_similarity(emb_a, emb_b)
        positive_scores.append(score)

    # 2. Process Negative Samples
    print(f"Processing {len(negative_samples)} negative samples...")
    for text_a, text_b in negative_samples:
        # Generate [CLS] embeddings for both texts
        emb_a = get_cls_embedding(text_a, model, tokenizer)
        emb_b = get_cls_embedding(text_b, model, tokenizer)
        
        # Calculate similarity and store
        score = calculate_cosine_similarity(emb_a, emb_b)
        negative_scores.append(score)

    # 3. Calculate Averages and Distance
    
    # Calculate average scores (using np.mean for easy handling of empty lists)
    avg_positive = np.mean(positive_scores) if positive_scores else 0.0
    avg_negative = np.mean(negative_scores) if negative_scores else 0.0
    
    # Calculate the distance between the two group means (simple absolute difference)
    distance = abs(avg_positive - avg_negative)
    # euclideadn distance can also be used for multi-dimensional embeddings
    #       File "/home/siddig/hpc/driver-license/hugging-drive/eval_legal_qa_generation.py", line 136, in evaluate_similarity
    #     euclidean_distance = np.linalg.norm(np.array(positive_scores) - np.array(negative_scores))
    # ValueError: operands could not be broadcast together with shapes (1957,) (1895,)
    euclidean_distance = np.linalg.norm(np.array([avg_positive]) - np.array([avg_negative]))

    print("\n--- Results Summary ---")
    print(f"Avg Positive Similarity: {avg_positive:.4f}")
    print(f"Avg Negative Similarity: {avg_negative:.4f}")
    print(f"Group Mean Distance: {distance:.4f}")
    print(f"Euclidean Distance: {euclidean_distance:.4f}")
    
    return positive_scores, negative_scores, avg_positive, avg_negative, euclidean_distance, distance

# plot box plots of the similarity scores with seaborn 
def plot_similarity_scores(pos_scores, neg_scores, plot_path=None):
    """
    Plots box plots of similarity scores for positive and negative samples using seaborn.
    
    Args:
        pos_scores (list of float): Similarity scores for positive samples.
        neg_scores (list of float): Similarity scores for negative samples.
        plot_path (str, optional): If provided, saves the plot to this path.
    """
    # Prepare data for plotting
    data = {
        'Similarity Score': pos_scores + neg_scores,
        'Type': ['Positive'] * len(pos_scores) + ['Negative'] * len(neg_scores)
    }
    df = pd.DataFrame(data)

    # Create box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Type', y='Similarity Score', data=df)
    plt.title('Similarity Scores for Positive and Negative Samples')
    plt.ylabel('Cosine Similarity Score')
    plt.xlabel('Sample Type')
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.show()
    if plot_path:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    
# save results into json file
def save_results_to_json(pos_scores, neg_scores, avg_pos, avg_neg, eucl_dist, dist, save_path):
    # avoid this one TypeError: Object of type float32 is not JSON serializable
    pos_scores = [float(score) for score in pos_scores]
    neg_scores = [float(score) for score in neg_scores]
    avg_pos = float(avg_pos)
    avg_neg = float(avg_neg)
    eucl_dist = float(eucl_dist)
    dist = float(dist)

    results = {
        "positive_scores": pos_scores,
        "negative_scores": neg_scores,
        "average_positive": avg_pos,
        "average_negative": avg_neg,
        "euclidean_distance": eucl_dist,
        "group_mean_distance": dist
    }
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")

### Threshold selection function to classify generated answers as correct or incorrect based on similarity scores
def find_optimal_threshold_roc(correct, incorrect):
    """
    Find threshold that maximizes TPR + TNR (Youden's J statistic)
    """
    best_threshold = 0.5
    best_j_statistic = 0
    
    for threshold in np.linspace(0, 1, 1001):
        # True Positive Rate (correct classified as correct)
        tp = np.sum(correct >= threshold)
        tpr = tp / len(correct) if len(correct) > 0 else 0
        
        # True Negative Rate (incorrect classified as incorrect)
        tn = np.sum(incorrect < threshold)
        tnr = tn / len(incorrect) if len(incorrect) > 0 else 0
        
        # Youden's J statistic (maximizing TPR + TNR - 1) : https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        j_statistic = tpr + tnr - 1
        
        if j_statistic > best_j_statistic:
            best_j_statistic = j_statistic
            best_threshold = threshold
    
    return best_threshold, best_j_statistic

def find_f1_optimal_threshold(correct, incorrect):
    """
    Find threshold that maximizes F1-score
    """
    best_threshold = 0.5
    best_f1 = 0
    f_1_scores = []
    thresholds = []
    
    for threshold in np.linspace(0, 1, 1001):
        tp = np.sum(correct >= threshold)
        fp = np.sum(incorrect >= threshold)
        fn = np.sum(correct < threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        thresholds.append(threshold)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f_1_scores.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1 , f_1_scores, thresholds

def find_accuracy_optimal_threshold(correct, incorrect):
    """
    Find threshold that maximizes accuracy
    """
    best_threshold = 0.5
    best_accuracy = 0
    accuracies = []
    thresholds = []
    
    for threshold in np.linspace(0, 1, 1001):
        tp = np.sum(correct >= threshold)
        tn = np.sum(incorrect < threshold)
        accuracy = (tp + tn) / (len(correct) + len(incorrect)) if (len(correct) + len(incorrect)) > 0 else 0
        accuracies.append(accuracy)
        thresholds.append(threshold)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy , accuracies, thresholds

def find_percentile_threshold(correct, incorrect, percentile=90):
    """
    Set threshold at nth percentile of correct answers
    Ensures high recall for correct answers
    """
    threshold = np.percentile(correct, percentile)
    return threshold

def plot_threshold_analysis(thresholds, f1_scores, accuracies, plot_path=None):
    """
    Plots F1-score and Accuracy against thresholds.
    """
    plt.figure(figsize=(10, 5))
    
    # F1-score plot
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, f1_scores, label='F1-score', color='blue')
    plt.title('F1-score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1-score')
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, accuracies, label='Accuracy', color='green')
    plt.title('Accuracy vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    if plot_path:
        plt.savefig(plot_path)
        print(f"Threshold analysis plot saved to {plot_path}")

##### EVALUTION SCRIPT #####
# "output_file": "../data/synthetic-data/stvo/citations/qa_llama_highquality.jsonl",
# {"question": "Can more passengers be carried in a vehicle if there are fewer seats equipped with safety belts?", "answer": "No, it is strictly prohibited under section 21 of the Transportation of Persons act to transport any person beyond the number of seats equipped with safety belts.", "label": "yes", "quality_score": 0.8, "is_yes_no_question": true, "language": "en", "paragraph_id": "§ 21", "paragraph": "§ 21Transportation of persons", "sentence": "(1) In motor vehicles no more persons may be transported than there are seats equipped with safety jets. Unlike paragraph 1, motor vehicles for which safety jets are not required for all seats may be transported as many persons as there are seats. It is prohibited to carry persons1. on power carriers without a special seat,2. on train machines without a suitable seat or3. in residential attachments behind motor vehicles.", "sentence_length": 424, "generation_attempt": 1}
# write function to eval generated answers in file "../data/synthetic-data/stvo/citations/qa_llama_highquality.jsonl given sample above.
# the function evaluate the quality of the generated answers by comparing them to the correct answers using similarity scores from Legal-BERT embeddings.
# use avg_similarity score as threshold to classify the generated answers as correct or incorrect. return the accuracy of the classification.

def evaluate_generated_answers(generated_file, model, tokenizer, threshold, q_a_sim=True):
    """
    Evaluate the quality of generated answers by comparing them to correct answers using similarity scores.
    
    Args:
        generated_file (str): Path to the JSONL file containing generated answers.
        model (PreTrainedModel): The loaded Legal-BERT model.
        tokenizer (PreTrainedTokenizer): The loaded Legal-BERT tokenizer.
        threshold (float): Similarity score threshold to classify answers as correct or incorrect.
        
    Returns:
        float: Accuracy of the classification.
    """
    correct_count = 0
    total_count = 0
    
    with open(generated_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            question = entry['question']
            generated_answer = entry['answer']
            paragraph = entry["paragraph"]
            # label either yes or no 
            correct_label = entry['label']

            # Generate embeddings
            if q_a_sim:
                question_emb = get_cls_embedding(question, model, tokenizer)
                answer_emb = get_cls_embedding(generated_answer, model, tokenizer)
                similarity_score = calculate_cosine_similarity(question_emb, answer_emb)
            
            
            # Classify based on threshold
            predicted_label = 'yes' if similarity_score >= threshold else 'no'
            if predicted_label == correct_label:
                correct_count += 1
            total_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy

def evaluate_generated_pargraphs(generated_file, model, tokenizer, threshold):
    """
    Evaluate the quality of generated paragraphs by comparing them to correct paragraphs using similarity scores.
    
    Args:
        generated_file (str): Path to the JSONL file containing generated paragraphs.
        model (PreTrainedModel): The loaded Legal-BERT model.
        tokenizer (PreTrainedTokenizer): The loaded Legal-BERT tokenizer.
        threshold (float): Similarity score threshold to classify paragraphs as correct or incorrect.
        
    Returns:
        float: Accuracy of the classification.
    """
    correct_count = 0
    total_count = 0
    
    with open(generated_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            question = entry['question']
            answer = entry['answer']
            context = question + " " + answer
            generated_paragraph = entry['paragraph']
            label = entry['label']  # Assuming label contains the correct paragraph text
            # Generate embeddings
            question_emb = get_cls_embedding(context, model, tokenizer)
            paragraph_emb = get_cls_embedding(generated_paragraph, model, tokenizer)
            similarity_score = calculate_cosine_similarity(question_emb, paragraph_emb)
            # Classify based on threshold
            predicted_label = 'yes' if similarity_score >= threshold else 'no'
            if predicted_label == label:
                correct_count += 1
            total_count += 1
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Legal QA Generation")
    parser.add_argument("--real_qa_path", type=str,default="./data/test-sets/austrian-driving-license/fragen", help="Path to the dataset containing questions and answers.")
    parser.add_argument("--generated_file", type=str, default="./data/synthetic-data/stvo/citations/qa_llama_highquality.jsonl", help="Path to the JSONL file containing generated answers.")
    parser.add_argument("--save_path", type=str, default="./data/synthetic-data/stvo/citations/similarity_scores.json", help="Path to save the similarity results JSON.")
    parser.add_argument("--plot_path", type=str, default="./data/synthetic-data/stvo/citations/similarity_scores_boxplot.png", help="Path to save the similarity scores boxplot.")
    parser.add_argument("--language", type=str, default="en", help="Language of the questions and answers.")
    parser.add_argument("--model_name", type=str, default="nlpaueb/legal-bert-base-uncased", help="Pretrained Legal-BERT model name.")
    parser.add_argument("--cache_dir", type=str, default="/fast_storage/siddig/driving-license/HF_models", help="Cache directory for HuggingFace models.")
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_handler = DrivingLicenseDataHandler()
    questions_answers = data_handler.read_austrian_questions_answers(args.real_qa_path, language=args.language)
    print(f"Loaded {len(questions_answers)} questions and answers for Austrian.")

    positives, negatives = separate_correct_incorrect(questions_answers)
    print(f"Number of correct answers: {len(positives)}")
    print(f"Number of incorrect answers: {len(negatives)}")
    
    # samples for verification
    print("\nSample entries:")
    if positives:
        print("Sample correct answer entry:", positives[:3])
    if negatives:
        print("Sample incorrect answer entry:", negatives[:3])
    print("\n")

    # Load Legal-BERT model and tokenizer
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name,cache_dir=args.cache_dir)
    model.to(device)
    print("Loaded Legal-BERT model and tokenizer.")

    # Method A: Evaluate similarity
    pos_scores, neg_scores, avg_pos, avg_neg, eucl_dist, dist = evaluate_similarity(
        [(entry['question'], entry['answer']) for entry in positives[:]],
        [(entry['question'], entry['answer']) for entry in negatives[:]],
        model,
        tokenizer
    )   
    # Save results to JSON
    save_results_to_json(pos_scores, neg_scores, avg_pos, avg_neg, eucl_dist, dist, save_path=args.save_path)
    # Plot similarity scores
    # plot_similarity_scores(pos_scores, neg_scores, plot_path=args.plot_path)

    # Method B : Find optimal threshold using ROC analysis
    optimal_threshold, j_statistic = find_optimal_threshold_roc(np.array(pos_scores), np.array(neg_scores))
    print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f} with J Statistic: {j_statistic:.4f}")
    
    # Method C : Find optimal threshold using F1-score maximization
    optimal_threshold_f1, f1_score, f_1_scores, thresholds = find_f1_optimal_threshold(np.array(pos_scores), np.array(neg_scores))
    print(f"Optimal Threshold (F1-score): {optimal_threshold_f1:.4f} with F1 Score: {f1_score:.4f}")
    # Method D: Find optimal threshold using accuracy maximization
    optimal_threshold_acc, accuracy, accuracies, thresholds_acc = find_accuracy_optimal_threshold(np.array(pos_scores), np.array(neg_scores))
    print(f"Optimal Threshold (Accuracy): {optimal_threshold_acc:.4f} with Accuracy: {accuracy:.4f}")
    # Plot threshold analysis
    # plot_threshold_analysis(thresholds, f_1_scores, accuracies, plot_path=args.plot_path.replace('.png', '_threshold_analysis.png'))

    # Method E : Find optimal threshold using percentile-based thresholding
    percentile = 90
    percentile_threshold = find_percentile_threshold(np.array(pos_scores), np.array(neg_scores), percentile=percentile)
    print(f"Percentile-based Threshold ({percentile}th percentile): {percentile_threshold:.4f}")
    
    
    # Evaluate generated answers using the average similarity score as threshold
    accuracy_q_a = evaluate_generated_answers(
        args.generated_file,
        model,
        tokenizer,
        threshold=0.915,
        q_a_sim=True
    )
    print(f"Accuracy of generated answers (Q-A similarity) at optimal ROC threshold: {accuracy_q_a:.4f}")
    
    

    # # evaluate generated paragraphs using the average similarity score as threshold
    # accuracy_paragraphs = evaluate_generated_pargraphs(
    #     args.generated_file,
    #     model,
    #     tokenizer,
    #     threshold=optimal_threshold
    # )
    # print(f"Accuracy of generated paragraphs at optimal ROC threshold: {accuracy_paragraphs:.4f}")
    

    # save optimal threshold to json file
    threshold_results = {
            "model_name": model_name,
            "average_positive_cosine_similarity": float(avg_pos),
            "average_negative_cosine_similarity": float(avg_neg),
            "euclidean_distance": float(eucl_dist),
            "group_mean_distance": float(dist),
            "optimal_ROC_threshold": float(optimal_threshold),
            "youden_j_ROC_statistic": float(j_statistic),
            "optimal_threshold_f1": float(optimal_threshold_f1),
            "f1_score": float(f1_score),
            "optimal_threshold_accuracy": float(optimal_threshold_acc),
            "accuracy": float(accuracy),
            "percentile_threshold": float(percentile_threshold),
            "percentile": percentile,
            "generated_answers_accuracy_q_a_similarity": float(accuracy_q_a)
        }
    
    with open(args.save_path.replace('.json', '_threshold.json'), 'w') as f:
        json.dump(threshold_results, f, indent=4)
    print(f"Threshold results saved to {args.save_path.replace('.json', '_threshold.json')}")
