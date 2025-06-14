import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse

def load_model(model_path):
    """Load the Mistral model using pipeline."""
    generator = pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return generator

def load_detective_puzzles_dataset(csv_path):
    """Load the detective puzzles dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    return df

def prepare_prompt(mystery_text, answer_options):
    """Prepare the input prompt for the model."""
    prompt = (
        f"<s>[INST] Read the following murder mystery and choose the correct culprit from the given options.\n\n"
        f"Mystery: {mystery_text}\n\n"
        f"Options: {answer_options}\n\n"
        f"Please provide your answer in the following format:\n"
        f"Answer: [one option from the options]\n"
        f"Reasoning: [your step-by-step reasoning for why this is the correct answer] [/INST]"
    )
    return prompt

def get_model_prediction(generator, mystery_text, answer_options):
    """Get the model's prediction for a given text using pipeline."""
    prompt = prepare_prompt(mystery_text, answer_options)
    outputs = generator(
        prompt,
        max_new_tokens=100, #roy!! it is 20 by default
        do_sample=True,
        temperature=0.7, #randomness
        top_p=0.9, # only the top 90% of tokens are considered
        return_full_text=False # deletes the [INST] tags
    )
    
    response = outputs[0]['generated_text']
    print("Model response:", response.strip())
    
    answer = response.split("Answer:")[1].split("\n")[0].strip() if "Answer:" in response else response.strip()
    return answer

def evaluate_model(generator, df):
    """Evaluate the model on the dataset."""
    predictions = []
    true_labels = []
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mystery_text = row['mystery_text']
        true_label = row['answer'].strip()
        answer_options = row['answer_options']
        
        pred = get_model_prediction(generator, mystery_text, answer_options)
        predictions.append(pred)
        true_labels.append(true_label)
        results.append({
            'mystery_text': mystery_text,
            'answer_options': answer_options,
            'true_answer': true_label,
            'model_prediction': pred
        })
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_predictions.csv', index=False)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on detective puzzles dataset.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate on. If not provided, evaluate on all samples.')
    parser.add_argument('--model_path', type=str, default="saved_mistralai_model", help='Path to the saved model directory')
    args = parser.parse_args()

    print("Loading model...")
    generator = load_model(args.model_path)
    
    csv_path = "data/detective-puzzles.csv"
    print(f"Loading dataset from {csv_path} ...")
    df = load_detective_puzzles_dataset(csv_path)
    
    if args.num_samples is not None:
        df = df.head(args.num_samples)
        print(f"Evaluating on {args.num_samples} samples.")
    
    print("Evaluating model...")
    metrics = evaluate_model(generator, df)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main() 