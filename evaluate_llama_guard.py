import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse

def load_llama_guard_model(model_path):
    """Load the LLaMA Guard model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def load_detective_puzzles_dataset(csv_path):
    """Load the detective puzzles dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    return df

def prepare_prompt(mystery_text, answer_options):
    """Prepare the input prompt for LLaMA Guard."""
    prompt = (
        f"<s>[INST] Read the following murder mystery and choose the correct culprit from the given options.\n\n"
        f"Mystery: {mystery_text}\n\n"
        f"Options: {answer_options}\n\n"
        f"Please answer with one word from the options. [/INST]"
    )
    return prompt

def get_model_prediction(model, tokenizer, mystery_text, answer_options):
    """Get the model's prediction for a given text."""
    prompt = prepare_prompt(mystery_text,answer_options)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("Model answer:", response.strip())
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=50,  # Increased from 1 to allow for longer responses
    #         temperature=0.7,    # Add some randomness
    #         do_sample=True,     # Enable sampling
    #         top_p=0.9,         # Nucleus sampling
    #         pad_token_id=tokenizer.eos_token_id
    #     )
    
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("\nFull response:", response)  # Debug print
    # print("Prompt:", prompt)            # Debug print
    return response


def evaluate_model(model, tokenizer, df):
    """Evaluate the model on the dataset."""
    predictions = []
    true_labels = []
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mystery_text = row['mystery_text']
        true_label = row['answer'].strip()
        answer_options = row['answer_options']
        
        pred = get_model_prediction(model, tokenizer, mystery_text, answer_options)
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
    parser = argparse.ArgumentParser(description='Evaluate LLaMA Guard model on detective puzzles dataset.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate on. If not provided, evaluate on all samples.')
    args = parser.parse_args()

    model_path = "saved_llama_guard_model"
    print("Loading LLaMA Guard model...")
    model, tokenizer = load_llama_guard_model(model_path)
    
    csv_path = "data/detective-puzzles.csv"
    print(f"Loading dataset from {csv_path} ...")
    df = load_detective_puzzles_dataset(csv_path)
    
    if args.num_samples is not None:
        df = df.head(args.num_samples)
        print(f"Evaluating on {args.num_samples} samples.")
    
    print("Evaluating model...")
    metrics = evaluate_model(model, tokenizer, df)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main() 