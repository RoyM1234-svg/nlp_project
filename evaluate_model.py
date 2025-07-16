from typing import Optional
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score
from models.detective_model import DetectiveModel
from models.llama_cot_model import LLamaCotModel
from models.llama_detective_model import LLamaDetectiveModel
from models.deepseek_detective_model import DeepSeekR1DistillQwen1_5BDetectiveModel
from models.gemma3_detective_model import Gemma3DetectiveModel
from data_loaders.detective_data_loader import DetectiveDataLoader
from models.llama_final_answer_model import LLamaFinalAnswerModel


def calculate_accuracy(predictions, true_labels):
    """Calculate accuracy for the predictions."""
    # Convert to consistent format (handle potential string/int mismatches)
    pred_clean = [str(pred).strip().lower() for pred in predictions]
    true_clean = [str(true).strip().lower() for true in true_labels]
    
    accuracy = accuracy_score(true_clean, pred_clean)
    return accuracy

def generate_from_model(model, mysteries, suspects, cots=None, k=1):
    """
       Helper to create prompts, tokenize, and generate outputs from a model.

       Parameters:
       - model: The model with `create_prompt`, `tokenizer`, `generate_batch`.
       - mysteries: List of mystery texts.
       - suspects: List of suspects lists.
       - cots: Optional list of COTs for prompt creation.
       - k: Number of generations per prompt.

       Returns:
       - Generated outputs from the model.
       """

    prompts = [
        model.create_prompt(mystery, suspect_list, cot)
        for mystery, suspect_list, cot in zip(mysteries, suspects, cots)
    ]

    inputs = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model.generate_batch(inputs['input_ids'], inputs['attention_mask'], k=k)
    return outputs


def evaluate_model(cot_model: DetectiveModel, final_answer_model: DetectiveModel, csv_path: str,
                   num_samples: Optional[int], batch_size: int, k: int):
    df = pd.read_csv(csv_path)
    if num_samples:
        df = df.sample(num_samples)
    data_loader = DetectiveDataLoader(df, batch_size=batch_size, shuffle=False)

    results = []

    for batch in tqdm(data_loader, desc="Generating batch"):

        generated_cots = generate_from_model(
            cot_model,
            batch['mystery_texts'],
            batch['suspects'],
            k=k
        )

        final_answers = generate_from_model(
            final_answer_model,
            batch['mystery_texts'],
            batch['suspects'],
            cots=generated_cots,
            k=k
        )
        
        for i in range(len(final_answers)):
            results.append({
                'generated_cots': generated_cots[i],
                'predictions': final_answers[i],
                'true_labels': batch['true_labels'][i],
                'indices': batch['indices'][i],
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    print(f"Results saved to results.csv")
    
    # Calculate and display accuracy
    final_answers = [result['predictions'] for result in results]
    true_labels = [result['true_labels'] for result in results]
    
    accuracy = calculate_accuracy(final_answers, true_labels)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def evaluate_model_for_self_consistency(model: DetectiveModel, csv_path: str, num_samples: Optional[int], batch_size: int, k: int):
    """
    Generates k samples for each puzzle using batching and saves them to a detailed CSV.
    The output format is one row per generated sample.
    """
    df = pd.read_csv(csv_path)
    if num_samples:
        df = df.sample(num_samples)
    
    data_loader = DetectiveDataLoader(df, batch_size=batch_size, shuffle=False)

    all_samples_log = []

    for batch in tqdm(data_loader, desc="Generating Batches"):
        input_ids = batch['inputs']['input_ids']
        attention_mask = batch['inputs']['attention_mask']
        
        # This call returns BATCH_SIZE * k items
        all_full_responses, all_predictions = model.generate_batch(input_ids, attention_mask, k=k)
        
        # --- CORE LOGIC TO UN-FLATTEN THE BATCH ---
        
        actual_batch_size = len(batch['indices'])
        
        # Loop 1: Iterate through each original puzzle in the batch
        for i in range(actual_batch_size):
            # Loop 2: Iterate through the k samples for THAT puzzle
            for j in range(k):
                # This is the key to finding the correct item in the flat list
                flat_index = i * k + j
                
                # Build the row with exactly the columns you want
                sample_row = {
                    'puzzle_id': batch['indices'][i],
                    'sample_index': j + 1,  # e.g., 1, 2, 3... up to k
                    'mystery_text': batch['mystery_texts'][i],     # Requires change in DataLoader
                    'answer_options': batch['answer_options'][i], # Requires change in DataLoader
                    'true_labels': batch['true_labels'][i],
                    'predictions': all_predictions[flat_index],
                    'generated_cots': all_full_responses[flat_index]
                }
                all_samples_log.append(sample_row)

    # --- SAVE THE FINAL CSV ---
    
    results_df = pd.DataFrame(all_samples_log)
    
    # Reorder columns to match your exact specification
    final_columns = [
        'puzzle_id', 'sample_index', 'mystery_text',
        'answer_options', 'true_labels', 'predictions', 'generated_cots'
    ]
    results_df = results_df[final_columns]
    
    results_df.to_csv("self_consistency_results.csv", index=False)
    print(f"\nSuccessfully generated {len(results_df)} samples.")
    print(f"Detailed results saved to self_consistency_results.csv")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on detective puzzles dataset.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate on. If not provided, evaluate on all samples.')
    parser.add_argument('--model_path', type=str, default="saved_llama_model", help='Path to the saved model directory')
    parser.add_argument('--model_type', type=str, default="llama", choices=["llama", "deepseek", "gemma3"], help='Type of model to use: llama/ deepseek/ gemma3')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--csv_path", type=str, default="data/detective-puzzles.csv")
    parser.add_argument("--k", type=int, default=10, help='Number of different outputs to generate per puzzle.')
    args = parser.parse_args()

    # Create the appropriate model based on the model_type argument
    if args.model_type == "llama":
        cot_model = LLamaCotModel(args.model_path)
        final_answer_model = LLamaFinalAnswerModel(args.model_path)
    # elif args.model_type == "deepseek":
    #     model = DeepSeekR1DistillQwen1_5BDetectiveModel(is_quantized=True)
    # elif args.model_type == "gemma3":
    #     model = Gemma3DetectiveModel(is_quantized=False)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    evaluate_model(cot_model, final_answer_model, args.csv_path, args.num_samples, args.batch_size, args.k)
    

if __name__ == "__main__":
    main() 