#!/usr/bin/env python3
"""
Script to evaluate the verifier model on golden Chain-of-Thought reasoning
from the detective-puzzles.csv dataset.
"""

import pandas as pd
import numpy as np
import sys
import os
import re
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.verifier_model import VerifierModel

VERIFIER_QUESTION = "Who is the guilty suspect?"

def extract_suspects_from_options(answer_options: str) -> str:
    if pd.isna(answer_options) or not answer_options:
        return ""
    pattern = r'\([a-d]\)\s*([^;]+)(?:;|$)'
    matches = re.findall(pattern, answer_options)
    
    suspects = [name.strip() for name in matches if name.strip()]
    return ", ".join(suspects)

def format_input_for_verifier(mystery_text: str, suspects: str, outcome: str) -> str:
    text = (
        str(mystery_text)
        + "\nSuspects: " + str(suspects)
        + "\nQuestion: " + VERIFIER_QUESTION
        + "\nChain of Thought: " + str(outcome)
    )
    return text

def load_and_preprocess_data(csv_path: str) -> Tuple[List[str], List[str]]:
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} cases")
    
    formatted_texts = []
    case_names = []
    skipped_cases = 0
    
    for idx, row in df.iterrows():
        if pd.isna(row['mystery_text']) or pd.isna(row['outcome']) or pd.isna(row['answer_options']):
            skipped_cases += 1
            continue
            
        suspects = extract_suspects_from_options(row['answer_options'])
        
        if not suspects:
            skipped_cases += 1
            continue
            
        formatted_text = format_input_for_verifier(
            row['mystery_text'], 
            suspects, 
            row['outcome']
        )
        
        formatted_texts.append(formatted_text)
        case_names.append(row['case_name'])
    
    print(f"Processed {len(formatted_texts)} cases successfully")
    if skipped_cases > 0:
        print(f"Skipped {skipped_cases} cases due to missing data")
    
    return formatted_texts, case_names

def evaluate_verifier_on_golden_cots(
    model_path: str,
    csv_path: str,
    batch_size: int = 8
) -> Tuple[List[float], float, List[str]]:

    print("Loading verifier model...")
    verifier = VerifierModel(model_path)
    
    formatted_texts, case_names = load_and_preprocess_data(csv_path)
    
    if not formatted_texts:
        print("No valid cases found for evaluation!")
        return [], 0.0, []
    
    print(f"Evaluating verifier on {len(formatted_texts)} golden CoTs...")
    
    all_scores = []
    
    for i in range(0, len(formatted_texts), batch_size):
        batch_texts = formatted_texts[i:i + batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(formatted_texts) + batch_size - 1)//batch_size}")
        
        batch_scores = verifier.predict_prob_correct(batch_texts)
        all_scores.extend(batch_scores)
    
    average_score = np.mean(all_scores)
    
    print(f"\nEvaluation Results:")
    print(f"Total cases evaluated: {len(all_scores)}")
    print(f"Average verifier score: {average_score:.4f}")
    print(f"Standard deviation: {np.std(all_scores):.4f}")
    print(f"Min score: {min(all_scores):.4f}")
    print(f"Max score: {max(all_scores):.4f}")
    
    
    return all_scores, average_score, case_names

def save_detailed_results(
    scores: List[float], 
    case_names: List[str], 
    output_path: str
):
    results_df = pd.DataFrame({
        'case_name': case_names,
        'verifier_score': scores
    })
    
    results_df.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")

def main():
    model_path = "musr/verifier_model"
    csv_path = "data/detective-puzzles.csv"
    output_path = "analysis/verifier_golden_cot_evaluation_results.csv"
    
    if not os.path.exists(model_path):
        print(f"Error: Verifier model not found at {model_path}")
        print("Please train the verifier model first or provide correct path.")
        return
    
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return
    
    try:
        scores, average_score, case_names = evaluate_verifier_on_golden_cots(
            model_path, 
            csv_path,
            batch_size=8
        )
        
        if scores:
            save_detailed_results(scores, case_names, output_path)
            
            print(f"\n{'='*50}")
            print(f"FINAL RESULT:")
            print(f"Average verifier score on golden CoTs: {average_score:.4f}")
            print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 