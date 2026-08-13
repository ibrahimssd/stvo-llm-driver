import json
import os
import re
from typing import List, Dict, Any, Tuple
import logging
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_sections_from_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n+', '\n', text)

    # Locate section headers
    tenor_match = re.search(r'\bTenor\b', text, re.IGNORECASE)
    tatbestand_match = re.search(r'\bTatbestand\b|T\s*a\s*t\s*b\s*e\s*s\s*t\s*a\s*n\s*d', text, re.IGNORECASE)
    gruende_match = re.search(r'\bEntscheidungsgr[üu]nde\b', text, re.IGNORECASE)


    tenor = tatbestand = gruende = ""

    if tenor_match and tatbestand_match:
        tenor = text[tenor_match.end():tatbestand_match.start()].strip()

    if tatbestand_match and gruende_match:
        tatbestand = text[tatbestand_match.end():gruende_match.start()].strip()

    if gruende_match:
        gruende = text[gruende_match.end():].strip()

    # Legal references (e.g., §§ 3, 4 UWG)
    legal_refs = re.findall(r'§{1,2}\s?[0-9]+[a-z]*(?:\s*Abs\.\s*\d+)?(?:\s*S\.\s*\d+)?(?:\s*Nr\.\s*\d+)?(?:\s*[a-z])?(?:\s*[A-Z]+)?', text)
    legal_refs = [ref.strip() for ref in legal_refs if ref.strip()]


    return {
        "tenor": tenor,
        "tatbestand": tatbestand,
        "entscheidungsgruende": gruende,
        "legal_references": legal_refs
    }


if __name__ == "__main__":
    # file_path = "../data/cases.jsonl"
    # output = []

    # with open(file_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         case = json.loads(line)
    #         logging.info(f"Processing case ID: {case['id']}")
    #         sections = extract_sections_from_content(case.get("content", ""))

    #         # if one filed empty skip
    #         if not sections["tenor"] or not sections["tatbestand"] or not sections["entscheidungsgruende"] or not sections["legal_references"]:
    #             logging.warning(f"Skipping case ID {case['id']} due to missing sections.")
    #             continue

    #         if case["id"] == 325566 or case["id"] == 323770 or case["id"] == 343880 or case["id"] == 343848:
    #             logging.info(f"📜 Tenor: {sections['tenor']}...")
    #             logging.info(f"📘 Tatbestand: {sections['tatbestand']}...")
    #             logging.info(f"🧠 Entscheidungsgründe: {sections['entscheidungsgruende']}...")
    #             logging.info(f"📚 Legal References: {sections['legal_references']}")
    #             logging.info("--------------------------------------------------")
                

    #         output.append({
    #             "id": case["id"],
    #             "tenor": sections["tenor"], # decision
    #             "tatbestand": sections["tatbestand"], # facts or case summary
    #             "entscheidungsgruende": sections["entscheidungsgruende"], # reasoning
    #             "legal_references": sections["legal_references"] # legal references
    #         })

    output_file_path = "./formatted_cases.jsonl"
    # with open(output_file_path, 'w', encoding='utf-8') as outfile:
    #     for result in output:
    #         json.dump(result, outfile, ensure_ascii=False)
    #         outfile.write('\n')

    # logging.info(f"✅ Processed results saved to {output_file_path}")

    # log examples 
    # with open(output_file_path, 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         if i < 5:
    #             case = json.loads(line)
    #             print(f"Case ID: {case['id']}")
    #             print(f"📜 Tenor: {case['tenor']}"
    #                     f"\n📘 Tatbestand: {case['tatbestand']}"
    #                   f"\n🧠 Entscheidungsgründe: {case['entscheidungsgruende']}"
    #                   f"\n📚 Legal References: {case['legal_references']}" )
    #             print("--------------------------------------------------")
    # number of samples in formatted cases
    num_samples = 0
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            num_samples += 1
    logging.info(f"Number of formatted cases: {num_samples}")
    # save some samples to share via email 
    samples = []
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 20:
                case = json.loads(line)
                samples.append(case)

    # save samples to a JSONL file
    samples_file_path = "./samples_formatted_cases_BGB.jsonl"
    with open(samples_file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
        logging.info(f"✅ Samples saved to {samples_file_path}")