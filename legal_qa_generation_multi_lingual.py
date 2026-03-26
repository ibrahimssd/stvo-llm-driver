import argparse
import json
import os
import torch
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from pathlib import Path
import gc
from datetime import datetime
from collections import defaultdict
import random


# --- Generation Parameters ---
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.5
DO_SAMPLE = True
TOP_P = 0.95
REPETITION_PENALTY = 1.2
BATCH_SIZE = 1

# --- Data Generation Parameters ---
DEFAULT_NUM_SAMPLES = 1000
QA_PAIRS_PER_SENTENCE = 2

# --- Quality Control Parameters ---
MIN_QUESTION_LENGTH = 15
MIN_ANSWER_LENGTH = 20
MAX_ANSWER_LENGTH = 500
QUALITY_SCORE_THRESHOLD = 0.6


# --- Language-specific configurations ---
LANGUAGE_CONFIG = {
    "de": {
        "name": "Deutsch",
        "legal_terms": [
            "regulation", "anforderung", "law", "muss", "soll", "verboten", "erlaubt",
            "erlaubnis", "lizenz", "einhaltung", "verstoß", "strafe", "ausnahme",
            "bedingung", "vorschrift", "artikel", "abschnitt", "fahrer", "fahrzeug",
            "straße", "verkehr", "geschwindigkeit", "entfernung", "signal", "zeichen",
            "verkehrssicherheit", "verhalten", "vorsicht", "rücksicht", "fahrbahn",
            "fußgänger", "radfahrer", "kraftfahrzeug", "kraftrad", "lkw", "pkw",
            "haltlinie", "überholverbot", "rechtsabbiegeverbot", "vorfahrt",
            "fahrspurwechsel", "parkplatz", "halteverbot", "einbahnstraße"
        ],
        "yes_no_patterns": [
            r'^(ist|sind|war|waren|darf|dürfen|durfte|kann|könnte|wird|würde|soll|sollte|muss|musste|hat|haben|hatte|hatten)\s',
        ],
        "generic_patterns": [
            r'^\s*was\s+ist\s+', 
            r'^\s*wie\s+kann\s+ich\s+', 
            r'^\s*warum\s+ist\s+'
        ]
    },
    "en": {
        "name": "English",
        "legal_terms": [
            "regulation", "requirement", "law", "must", "shall", "prohibited", "allowed",
            "permission", "license", "comply", "violation", "penalty", "exception",
            "condition", "provision", "article", "section", "driver", "vehicle",
            "road", "traffic", "speed", "distance", "signal", "sign",
            "traffic safety", "behavior", "caution", "consideration", "roadway",
            "pedestrian", "cyclist", "motor vehicle", "motorcycle", "truck", "car",
            "stop line", "no overtaking", "no right turn", "right of way",
            "lane change", "parking lot", "no parking", "one-way street"
        ],
        "yes_no_patterns": [
            r'^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had|must|may)\s',
        ],
        "generic_patterns": [
            r'^\s*what\s+is\s+', 
            r'^\s*how\s+do\s+i\s+', 
            r'^\s*why\s+is\s+'
        ]
    }
}


def setup_logging(output_dir: str):
    """Setup logging configuration with timestamped log files."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"citation_qa_generation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")


def get_legal_domain_prompt(
    sentence_text: str,
    paragraph_full_text: str,
    pairs_per_sentence: int,
    domain_type: str = "traffic_law",
    language: str = "de"
) -> str:
    """
    Generate domain-specific prompts for legal Q&A generation in specified language.
    
    Args:
        sentence_text: The legal sentence
        paragraph_full_text: Context paragraph
        pairs_per_sentence: Number of pairs needed
        domain_type: Type of legal domain
        language: Target language ('de' for German, 'en' for English)
    
    Returns:
        Optimized prompt for legal domain in specified language
    """
    num_positive = pairs_per_sentence // 2 + pairs_per_sentence % 2
    num_negative = pairs_per_sentence // 2
    
    # German prompts
    de_domain_instructions = {
        "traffic_law": {
            "context": "Verkehrsrecht und Straßenverkehrssicherheit",
            "requirements": [
                "- Fragen sollten rechtliche Verpflichtungen, Verbote oder Ausnahmen klären",
                "- Fragen MÜSSEN offen sein, KEINE Ja/Nein-Fragen",
                "- Fragen DÜRFEN NICHT auf Bilder, Texte oder Schilder verweisen oder vorherigen Kontext voraussetzen",
                "- Antworten müssen spezifische Bedingungen oder Umstände referenzieren",
                "- Negative Beispiele sollten plausible Verstöße oder Fehlinterpretationen sein",
                "- Fokus auf praktische Anwendung und Compliance-Szenarien"
            ]
        },
        "regulatory": {
            "context": "Rechtliche Behördenvorschriften und Compliance",
            "requirements": [
                "- Fragen sollten das Verständnis von Behördenvorschriften testen",
                "- Fragen MÜSSEN offen sein, KEINE Ja/Nein-Fragen",
                "- Fragen DÜRFEN NICHT auf Bilder, Texte oder Schilder verweisen oder vorherigen Kontext voraussetzen",
                "- Antworten müssen rechtlich korrekt und zitierkompatibel sein",
                "- Negative Beispiele sollten häufige Fehlinterpretationen sein",
                "- Bedingte Anforderungen und Ausnahmen einbeziehen"
            ]
        },
        "contract": {
            "context": "Vertragsrecht und rechtliche Verpflichtungen",
            "requirements": [
                "- Fragen sollten Rechte, Pflichten und Verpflichtungen klären",
                "- Fragen MÜSSEN offen sein, KEINE Ja/Nein-Fragen",
                "- Fragen DÜRFEN NICHT auf Bilder, Texte oder Schilder verweisen oder vorherigen Kontext voraussetzen",
                "- Antworten müssen spezifische Vertragsbedingungen referenzieren",
                "- Negative Beispiele sollten rechtlich plausibel aber falsch sein",
                "- Fokus auf praktische Umsetzung und Grenzfälle"
            ]
        }
    }
    
    # English prompts
    en_domain_instructions = {
        "traffic_law": {
            "context": "Traffic regulation and road safety law",
            "requirements": [
                "- Questions should clarify legal obligations, prohibitions, or exceptions",
                "- Questions MUST be open-ended, NOT yes/no questions",
                "- Questions MUST NOT reference images or texts or signs or assume prior context",
                "- Answers must reference specific legal conditions or circumstances",
                "- Negative examples should be plausible violations or misinterpretations",
                "- Focus on practical application and compliance scenarios",
                
                

            ]
        },
        "regulatory": {
            "context": "Legal regulatory requirements and compliance",
            "requirements": [
                "- Questions should test understanding of regulatory requirements",
                "- Questions MUST be open-ended, NOT yes/no questions",
                "- Questions MUST NOT reference images or texts or signs or assume prior context",
                "- Answers must be legally accurate and citation-compatible",
                "- Negative examples should be common misconceptions",
                "- Include conditional requirements and exceptions",
                

            ]
        },
        "contract": {
            "context": "Contract law and legal obligations",
            "requirements": [
                "- Questions should clarify rights, duties, and obligations",
                "- Questions MUST be open-ended, NOT yes/no questions",
                "- Questions MUST NOT reference images or texts or signs or assume prior context",
                "- Answers must reference specific contractual terms or conditions",
                "- Negative examples should be legally plausible but incorrect",
                "- Focus on practical implementation and edge cases",
                
            ]
        }
    }
    
    # Select language instructions
    if language == "de":
        instructions = de_domain_instructions
        domain_info = instructions.get(domain_type, instructions["traffic_law"])
        
        prompt = f"""Du bist ein Experte für {domain_info["context"]}.

Deine Aufgabe: Erstelle genau {pairs_per_sentence} Frage-Antwort-Paare basierend auf dem Rechtstext unten im JSON-Format.

KRITISCHE ANFORDERUNGEN:
- Erstelle {num_positive} KORREKTE(s) Paar(e) mit Label "ja", wobei die Antwort faktisch KORREKT ist
- Erstelle {num_negative} FALSCHE(s) Paar(e) mit Label "nein", wobei die Antwort plausibel aber rechtlich FALSCH ist
{chr(10).join(domain_info['requirements'])}

QUALITÄTSSTANDARDS:
- Fragen MÜSSEN offen und spezifisch sein (keine Ja/Nein-Fragen)
- Fragen DÜRFEN NICHT auf Bilder, externe Dokumente oder vorherigen Kontext anspielen
- Fragen sollten 15-40 Wörter lang sein
- Antworten müssen vollständig und detailliert sein (20-100 Wörter)
- Antworten sollten in einfacher, klarer Sprache aber rechtlich präzise sein
- Jede Antwort muss eigenständig verständlich sein

QUELLTEXTANFORDERUNGEN:
- Für "ja" Label: Antwort muss vom bereitgestellten Rechtstext direkt gestützt sein
- Für "nein" Label: Antwort sollte plausibel sein aber den Rechtstext widersprechen
- Stelle sicher, dass Labels konsistent mit der Antwortgenauigkeit sind

QUELLTEXT:
Rechtssatz: "{sentence_text}"

Kontext (umgebender Absatz): "{paragraph_full_text}"

AUSGABEANWEISUNGEN:
1. Gib NUR gültiges JSON-Array-Format zurück
2. Jedes Objekt muss genau drei Felder haben: "frage", "antwort", "label"
3. Alle Felder müssen Strings sein
4. Keine Erklärungen, Markdown oder Code-Blöcke
5. Stelle sicheres JSON-Escaping für alle Sonderzeichen sicher

Erwartetes Ausgabeformat:
[
  {{"frage": "Was muss ein Fahrer beim Nähern an eine rote Ampel tun?", "antwort": "Ein Fahrer muss verlangsamen und...", "label": "ja"}},
  {{"frage": "Darf ein Fahrer bei Rot fahren wenn...", "antwort": "Dies ist nicht erlaubt unter...", "label": "nein"}}
]"""
    else:  # English
        instructions = en_domain_instructions
        domain_info = instructions.get(domain_type, instructions["traffic_law"])
        
        prompt = f"""You are a legal expert specializing in {domain_info["context"]}.

Your task: Generate exactly {pairs_per_sentence} high-quality question-answer pairs based on the legal text below.

CRITICAL REQUIREMENTS:
- Generate {num_positive} CORRECT pair(s) with label "yes" - factually accurate based on the legal text
- Generate {num_negative} INCORRECT pair(s) with label "no" - plausible but legally incorrect
{chr(10).join(domain_info['requirements'])}

QUALITY STANDARDS:
- Questions MUST be open-ended and specific (avoid yes/no questions)
- Questions MUST NOT reference images, external documents, or assume prior context
- Questions should be 15-40 words long
- Answers must be complete, detailed sentences (20-100 words)
- Answers should be in simple, clear language but legally precise
- Each answer must stand alone and be understandable without the question

LEGAL TEXT REQUIREMENTS:
- For "yes" label: Answer must be directly supported by the provided legal text
- For "no" label: Answer should be plausible but contradicts the legal text or introduces incorrect details
- Ensure labels are consistent with answer accuracy

SOURCE TEXT:
Legal Sentence: "{sentence_text}"

Context (surrounding paragraph): "{paragraph_full_text}"

OUTPUT INSTRUCTIONS:
1. Return ONLY valid JSON array format
2. Each object must have exactly three fields: "question", "answer", "label"
3. All fields must be strings
4. Do not include explanations, markdown, or code blocks
5. Ensure valid JSON escaping for all special characters

Expected output format:
[
  {{"question": "What is the specific requirement regarding...", "answer": "According to the regulation...", "label": "yes"}},
  {{"question": "Under what conditions must a driver...", "answer": "This requirement does not apply when...", "label": "no"}}
]"""
    
    return prompt


def extract_json_from_response(response_text: str, expected_pairs: int = 2, language: str = "de") -> List[Dict[str, Any]]:
    """
    Extract JSON array from model response with improved legal-specific parsing.
    Supports both German and English field names.
    
    Args:
        response_text: Raw model output
        expected_pairs: Expected number of Q&A pairs
        language: Language used ('de' or 'en')
        
    Returns:
        List of extracted Q&A dictionaries with normalized field names
    """
    # Define field names based on language
    if language == "de":
        question_key = "frage"
        answer_key = "antwort"
        label_key = "label"
    else:
        question_key = "question"
        answer_key = "answer"
        label_key = "label"
    
    # Strategy 1: Extract complete JSON array
    try:
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_string = json_match.group(0)
            parsed = json.loads(json_string)
            if isinstance(parsed, list):
                # Normalize field names to English for consistency
                normalized = []
                for item in parsed[:expected_pairs]:
                    normalized_item = {
                        "question": item.get(question_key) or item.get("question") or "",
                        "answer": item.get(answer_key) or item.get("antwort") or item.get("answer") or "",
                        "label": item.get(label_key) or item.get("label") or ""
                    }
                    if normalized_item["question"] and normalized_item["answer"]:
                        normalized.append(normalized_item)
                if normalized:
                    return normalized
    except (AttributeError, json.JSONDecodeError) as e:
        logging.debug(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Extract individual JSON objects
    try:
        # Support both German and English field names
        pattern = rf'\{{[^{{}}]*?"({question_key}|question)"[^{{}}]*?"({answer_key}|antwort|answer)"[^{{}}]*?"({label_key}|label)"[^{{}}]*?\}}'
        objects = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if len(objects) >= expected_pairs:
            # Extract full objects
            full_objects = re.findall(
                r'\{[^{}]*?"(?:frage|question)"[^{}]*?"(?:antwort|answer)"[^{}]*?"(?:label)"[^{}]*?\}',
                response_text,
                re.DOTALL | re.IGNORECASE
            )
            if len(full_objects) >= expected_pairs:
                json_string = '[' + ','.join(full_objects[:expected_pairs]) + ']'
                parsed = json.loads(json_string)
                normalized = []
                for item in parsed:
                    normalized_item = {
                        "question": item.get(question_key) or item.get("question") or "",
                        "answer": item.get(answer_key) or item.get("antwort") or item.get("answer") or "",
                        "label": item.get(label_key) or item.get("label") or ""
                    }
                    if normalized_item["question"] and normalized_item["answer"]:
                        normalized.append(normalized_item)
                if normalized:
                    return normalized
    except (AttributeError, json.JSONDecodeError) as e:
        logging.debug(f"Strategy 2 failed: {e}")
    
    # Strategy 3: Extract fields individually
    try:
        questions = re.findall(
            rf'"(?:{question_key}|question)"\s*:\s*"([^"]*)"',
            response_text,
            re.IGNORECASE
        )
        answers = re.findall(
            rf'"(?:{answer_key}|antwort|answer)"\s*:\s*"([^"]*)"',
            response_text,
            re.IGNORECASE
        )
        labels = re.findall(
            rf'"(?:{label_key}|label)"\s*:\s*(?:"([^"]*)"|([a-z]+))',
            response_text,
            re.IGNORECASE
        )
        
        if len(questions) >= expected_pairs and len(answers) >= expected_pairs:
            pairs = []
            for i in range(min(expected_pairs, len(questions), len(answers))):
                if i < len(labels):
                    label = labels[i][0] if labels[i][0] else labels[i][1]
                else:
                    label = "ja" if language == "de" else "yes" if i == 0 else ("nein" if language == "de" else "no")
                
                label = label.lower().strip()
                valid_labels = ["ja", "nein"] if language == "de" else ["yes", "no"]
                if label not in valid_labels:
                    label = "ja" if language == "de" else "yes" if i == 0 else ("nein" if language == "de" else "no")
                
                pairs.append({
                    "question": questions[i],
                    "answer": answers[i],
                    "label": label
                })
            return pairs
    except Exception as e:
        logging.debug(f"Strategy 3 failed: {e}")
    
    logging.warning(f"Could not extract valid JSON. Response preview: {response_text[:200]}...")
    return []


def calculate_quality_score(
    question: str,
    answer: str,
    sentence_text: str,
    label: str,
    language: str = "en"
) -> float:
    """
    Calculate a quality score for a Q&A pair (0-1) with language support.
    """
    score = 1.0
    
    # Question quality
    if len(question) < MIN_QUESTION_LENGTH:
        score -= 0.3
    elif len(question) > 200:
        score -= 0.2
    
    # Avoid generic questions (language-specific patterns)
    lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["en"])
    if any(re.match(p, question.lower()) for p in lang_config["generic_patterns"]):
        score -= 0.1
    
    # Answer quality
    if len(answer) < MIN_ANSWER_LENGTH:
        score -= 0.4
    elif len(answer) > MAX_ANSWER_LENGTH:
        score -= 0.2
    
    # Check for relevance (word overlap with source)
    source_words = set(sentence_text.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(source_words & answer_words) / max(len(source_words), 1)
    
    if overlap < 0.1 and label in ["ja", "yes"]:
        score -= 0.3
    
    if overlap > 0.8:
        score -= 0.15
    
    # Penalty for yes/no answers to yes/no questions
    if question.lower().endswith('?') and answer.lower() in ['ja', 'nein', 'yes', 'no', 'true', 'false']:
        score -= 0.4
    
    if answer.lower().startswith('i don\'t know') or answer.lower().startswith('ich weiß nicht'):
        score -= 0.5
    
    # Repetition check
    word_list = answer.lower().split()
    if len(word_list) > 5:
        avg_word_freq = len(word_list) / len(set(word_list))
        if avg_word_freq > 0.4:
            score -= 0.2
    
    return max(0.0, min(1.0, score))


def validate_qa_pair(
    pair: Dict[str, Any],
    sentence_text: str,
    language: str = "en",
    calculate_score: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Comprehensive validation for legal Q&A pairs with language support.
    """
    try:
        # Normalize field names from generation
        if "frage" in pair:
            pair["question"] = pair.get("frage", pair.get("question"))
        if "antwort" in pair:
            pair["answer"] = pair.get("antwort", pair.get("answer"))
        
        # Check required fields
        if not all(key in pair for key in ['question', 'answer', 'label']):
            return None
        
        question = str(pair['question']).strip()
        answer = str(pair['answer']).strip()
        label = str(pair['label']).lower().strip()
        
        # === LANGUAGE CONSISTENCY CHECK ===
        # check the language if the pair is english while the language is german and vice versa
        if language == "de":
            # Simple heuristic: check for presence of common English words
            common_english_words = ['the', 'is', 'and', 'of', 'to', 'a', 'in', 'that', 'it', 'on']
            if any(word in question.lower() for word in common_english_words) or \
               any(word in answer.lower() for word in common_english_words):
                logging.debug(f"Language inconsistency detected in German pair: {question}")
                return None
        
        # === LEGAL DOMAIN VALIDATION ===
        
        if len(question) < MIN_QUESTION_LENGTH:
            logging.debug(f"Question too short: {question}")
            return None
        
        if len(question) > 200:
            logging.debug(f"Question too long: {question}")
            return None
        
        if len(answer) < MIN_ANSWER_LENGTH:
            logging.debug(f"Answer too short: {answer}")
            return None
        
        if len(answer) > MAX_ANSWER_LENGTH:
            logging.debug(f"Answer too long: {answer}")
            return None
        
        # Normalize and validate label
        valid_labels = ["ja", "nein"] if language == "de" else ["yes", "no"]
        if language == "de" and label in ["yes", "no"]:
            label = "ja" if label == "yes" else "nein"
        elif language == "en" and label in ["ja", "nein"]:
            label = "yes" if label == "ja" else "no"
        
        if label not in valid_labels:
            logging.debug(f"Invalid label: {label}")
            return None
        
        # Avoid identical question and answer
        if question.lower() == answer.lower():
            logging.debug("Question and answer are identical")
            return None
        
        # Check for invalid yes/no question-answer pairs
        lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["en"])
        question_lower = question.lower()
        is_yes_no_question = any(
            re.match(pattern, question_lower, re.IGNORECASE)
            for pattern in lang_config["yes_no_patterns"]
        )
        
        if is_yes_no_question and answer.lower() in ['yes', 'no', 'true', 'false', 'ja', 'nein', 'wahr', 'falsch']:
            logging.debug(f"Yes/No question with yes/no answer rejected: {question}")
            return None
        
        # Legal-specific checks
        
        
        # 1. Avoid placeholder content
        placeholders = ['xxx', 'example', 'placeholder', '[...]', '...', 'beispiel', 'platzhalter']
        if any(p in question.lower() or p in answer.lower() for p in placeholders):
            logging.debug(f"Placeholder content detected: {question}")
            return None
        
        # 2. Ensure answer is not just a repeat of question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap_ratio = len(question_words & answer_words) / len(question_words | answer_words)
        
        if overlap_ratio > 0.7:
            logging.debug(f"Answer too similar to question (overlap: {overlap_ratio:.2f})")
            return None
        
        # 3. Check for legal terminology
        legal_terms = lang_config["legal_terms"]
        combined_text = (question + ' ' + answer).lower()
        has_legal_term = any(term in combined_text for term in legal_terms)
        
        if not has_legal_term:
            logging.debug(f"Insufficient legal terminology: {question}")
            return None
        

        # Clean up whitespace
        question = re.sub(r'\s+', ' ', question)
        answer = re.sub(r'\s+', ' ', answer)
        
        # Calculate quality score
        quality_score = calculate_quality_score(question, answer, sentence_text, label, language)
        
        if quality_score < QUALITY_SCORE_THRESHOLD:
            logging.debug(f"Quality score too low: {quality_score:.2f} (threshold: {QUALITY_SCORE_THRESHOLD})")
            return None
        
        # Return validated pair with metadata
        validated = {
            'question': question,
            'answer': answer,
            'label': label,
            'quality_score': round(quality_score, 3),
            'is_yes_no_question': is_yes_no_question,
            'language': language
        }
        
        # Preserve any additional fields
        for key in pair:
            if key not in validated:
                validated[key] = pair[key]
        
        return validated
        
    except Exception as e:
        logging.warning(f"Error validating Q&A pair: {e}")
        return None


def generate_qa_from_sentence(
    model,
    tokenizer,
    sentence_text: str,
    paragraph_id: str,
    paragraph_full_text: str,
    pairs_per_sentence: int = QA_PAIRS_PER_SENTENCE,
    domain_type: str = "traffic_law",
    language: str = "en",
    retry_count: int = 2
) -> List[Dict[str, Any]]:
    """
    Generate Q&A pairs for a legal sentence with language support.
    """
    
    instruction_prompt = get_legal_domain_prompt(
        sentence_text,
        paragraph_full_text,
        pairs_per_sentence,
        domain_type,
        language
    )
    
    messages = [{"role": "user", "content": instruction_prompt}]
    
    for attempt in range(retry_count):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(model.device)
            
            with torch.no_grad():
                generated_output_ids = model.generate(
                    inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    early_stopping=True
                )
            
            generated_token_ids = generated_output_ids[0, inputs.shape[1]:]
            response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
            
            # Extract and validate Q&A pairs
            qa_pairs = extract_json_from_response(response_text, pairs_per_sentence, language)
            
            validated_pairs = []
            for pair in qa_pairs:
                validated = validate_qa_pair(pair, sentence_text, language, calculate_score=True)
                if validated:
                    # Add citation metadata
                    validated['paragraph_id'] = paragraph_id
                    validated['paragraph'] = paragraph_full_text
                    validated['sentence'] = sentence_text
                    validated['sentence_length'] = len(sentence_text)
                    validated['generation_attempt'] = attempt + 1
                    validated_pairs.append(validated)
            
            # Return if we got sufficient pairs
            if len(validated_pairs) >= pairs_per_sentence * 0.5:
                return validated_pairs
        
        except Exception as e:
            logging.debug(f"Generation attempt {attempt + 1} failed: {e}")
            continue
    
    logging.warning(f"Failed to generate valid Q&A for sentence '{sentence_text[:50]}...'")
    return []


def collect_all_sentences(structured_data: List[Dict]) -> List[Tuple[str, str, str, str]]:
    """
    Collect all sentences from structured data with metadata.
    """
    all_sentences = []
    
    for category in structured_data:
        category_name = category.get('category', 'Unknown')
        
        for paragraph_data in category.get('paragraphs', []):
            paragraph_id = paragraph_data.get('paragraph_id', 'unknown')
            paragraph_full_text = paragraph_data.get('paragraph', '')
            
            for sentence_dict in paragraph_data.get('sentences', []):
                if isinstance(sentence_dict, dict):
                    sentence_key, sentence_text = list(sentence_dict.items())[0]
                else:
                    sentence_text = str(sentence_dict)
                
                # Skip very short or empty sentences
                if len(sentence_text.strip()) > 20:
                    all_sentences.append((
                        sentence_text,
                        paragraph_id,
                        paragraph_full_text,
                        category_name
                    ))
    
    return all_sentences


def save_dataset_info(output_path: Path, stats: Dict[str, Any]):
    """Save dataset generation statistics and metadata."""
    info_file = output_path.parent / f"{output_path.stem}_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logging.info(f"Dataset info saved to {info_file}")


def main():
    """Main execution function with language support."""
    
    parser = argparse.ArgumentParser(
        description="Generate high-quality citation-based Q&A pairs from legal text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--model_name', type=str,
                       default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="Hugging Face model name")
    parser.add_argument('--hf_cache_dir', type=str,
                       default="/temp/siddig/driving-license/HF_models",
                       help="Hugging Face cache directory")
    parser.add_argument('--hf_token', type=str,
                       default=None,
                       help="Hugging Face API token")
    
    # Language parameter
    parser.add_argument('--language', type=str,
                       default="en",
                       choices=['de', 'en'],
                       help="Target language for Q&A generation (de=German, en=English)")
    
    # Data generation parameters
    parser.add_argument('--num_samples', type=int,
                       default=DEFAULT_NUM_SAMPLES,
                       help="Target number of samples to generate")
    parser.add_argument('--pairs_per_sentence', type=int,
                       default=QA_PAIRS_PER_SENTENCE,
                       help="Number of Q&A pairs per sentence")
    parser.add_argument('--shuffle_sentences', action='store_true',
                       help="Shuffle sentences for diversity")
    parser.add_argument('--domain_type', type=str,
                       default="traffic_law",
                       choices=['traffic_law', 'regulatory', 'contract'],
                       help="Legal domain type for prompt optimization")
    
    # Quality parameters
    parser.add_argument('--min_quality_score', type=float,
                       default=QUALITY_SCORE_THRESHOLD,
                       help="Minimum quality score threshold (0-1)")
    parser.add_argument('--max_retries', type=int,
                       default=2,
                       help="Retries per sentence on generation failure")
    
    # Input/Output parameters
    parser.add_argument('--input_file', type=str,
                       default="../data/stvo/structured_legal_text.json",
                       help="Path to structured JSON legal text")
    parser.add_argument('--output_file', type=str,
                       default="../data/synthetic-data/citation_qa_data.jsonl",
                       help="Output file for Q&A pairs")
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_file).parent
    # setup_logging(output_dir)
    
    # Setup output path
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    language_name = LANGUAGE_CONFIG.get(args.language, {}).get("name", args.language)
    
    logging.info("=" * 80)
    logging.info("LEGAL Q&A GENERATION CONFIGURATION")
    logging.info("=" * 80)
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Language: {language_name} ({args.language})")
    logging.info(f"Target samples: {args.num_samples}")
    logging.info(f"Q&A pairs per sentence: {args.pairs_per_sentence}")
    logging.info(f"Domain type: {args.domain_type}")
    logging.info(f"Min quality score: {args.min_quality_score}")
    logging.info(f"Max retries: {args.max_retries}")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output file: {args.output_file}")
    logging.info("=" * 80)
    
    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.hf_cache_dir,
            trust_remote_code=False,
            token=args.hf_token
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logging.info("Set pad_token to eos_token")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.hf_cache_dir,
            device_map="auto",
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            token=args.hf_token
        )
        
        model.eval()
        logging.info(f"Model loaded on device: {model.hf_device_map}")
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
    
    # Load structured legal text
    try:
        logging.info(f"Reading structured legal text from {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        
        # Collect all sentences
        all_sentences = collect_all_sentences(structured_data)
        total_sentences = len(all_sentences)
        
        logging.info(f"Total sentences available: {total_sentences}")
        
        if total_sentences == 0:
            logging.error("No sentences found in input file")
            return
        
        # Shuffle if requested
        if args.shuffle_sentences:
            random.shuffle(all_sentences)
            logging.info("Sentences shuffled for diversity")
        
        # Generate Q&A pairs
        generated_count = 0
        label_counts = {'ja': 0, 'nein': 0} if args.language == "de" else {'yes': 0, 'no': 0}
        category_counts = defaultdict(int)
        paragraph_counts = defaultdict(int)
        quality_scores = []
        failed_sentences = 0
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            with tqdm(total=args.num_samples, desc=f"Generating {language_name} Q&A") as pbar:
                
                for idx, (sentence_text, paragraph_id, paragraph_text, category) in enumerate(all_sentences):
                    if generated_count >= args.num_samples:
                        break
                    
                    try:
                        qa_pairs = generate_qa_from_sentence(
                            model,
                            tokenizer,
                            sentence_text,
                            paragraph_id,
                            paragraph_text,
                            args.pairs_per_sentence,
                            args.domain_type,
                            args.language,
                            args.max_retries
                        )
                        
                        if not qa_pairs:
                            failed_sentences += 1
                        
                        # Write pairs and update statistics
                        for pair in qa_pairs:
                            if generated_count >= args.num_samples:
                                break
                            
                            outfile.write(json.dumps(pair, ensure_ascii=False) + '\n')
                            generated_count += 1
                            label_counts[pair['label']] += 1
                            category_counts[category] += 1
                            paragraph_counts[paragraph_id] += 1
                            quality_scores.append(pair.get('quality_score', 0))
                            pbar.update(1)
                        
                        # Periodic flush and memory cleanup
                        if generated_count % 100 == 0:
                            outfile.flush()
                            os.fsync(outfile.fileno())
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logging.error(f"Error processing sentence {idx}: {e}")
                        failed_sentences += 1
        
        # Calculate statistics
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        stats = {
            'total_samples': generated_count,
            'language': args.language,
            'language_name': language_name,
            'label_distribution': dict(label_counts),
            'balance_ratio': label_counts[list(label_counts.keys())[0]] / max(label_counts[list(label_counts.keys())[1]], 1),
            'category_distribution': dict(category_counts),
            'unique_paragraphs': len(paragraph_counts),
            'sentences_processed': total_sentences,
            'sentences_failed': failed_sentences,
            'success_rate': ((total_sentences - failed_sentences) / total_sentences * 100) if total_sentences > 0 else 0,
            'quality_metrics': {
                'average_score': round(avg_quality_score, 3),
                'min_score': round(min(quality_scores), 3) if quality_scores else 0,
                'max_score': round(max(quality_scores), 3) if quality_scores else 0,
            },
            'model_name': args.model_name,
            'domain_type': args.domain_type,
            'input_file': args.input_file,
            'output_file': str(output_path),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        save_dataset_info(output_path, stats)
        
        # Log final statistics
        logging.info("=" * 80)
        logging.info("GENERATION COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Total samples generated: {generated_count}")
        logging.info(f"Language: {language_name}")
        logging.info(f"Label distribution: {dict(label_counts)}")
        logging.info(f"Average quality score: {avg_quality_score:.3f}")
        logging.info(f"Unique paragraphs cited: {len(paragraph_counts)}")
        logging.info(f"Success rate: {stats['success_rate']:.1f}%")
        logging.info(f"Output saved to: {output_path}")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error(f"Error during generation: {e}", exc_info=True)
    
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleanup completed")


if __name__ == "__main__":
    main()