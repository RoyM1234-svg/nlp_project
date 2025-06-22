from typing import Optional
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score
from models.detective_model import DetectiveModel
from models.llama_detective_model import LLamaDetectiveModel
from models.deepseek_detective_model import DeepSeekR1DistillQwen1_5BDetectiveModel
from models.gemma3_detective_model import Gemma3DetectiveModel
from data_loaders.detective_data_loader import DetectiveDataLoader


def calculate_accuracy(predictions, true_labels):
    """Calculate accuracy for the predictions."""
    # Convert to consistent format (handle potential string/int mismatches)
    pred_clean = [str(pred).strip().lower() for pred in predictions]
    true_clean = [str(true).strip().lower() for true in true_labels]
    
    accuracy = accuracy_score(true_clean, pred_clean)
    return accuracy


def evaluate_model(model: DetectiveModel, csv_path: str, num_samples: Optional[int], batch_size: int):
    df = pd.read_csv(csv_path)
    if num_samples:
        df = df.sample(num_samples)
    data_loader = DetectiveDataLoader(model, df, batch_size=batch_size, shuffle=False)

    results = []

    for batch in tqdm(data_loader, desc="Generating batch"):
        print(batch['inputs']['input_ids'].shape)
        input_ids = batch['inputs']['input_ids']
        attention_mask = batch['inputs']['attention_mask']
        generated_cots, predictions = model.generate_batch(input_ids, attention_mask)
        for i in range(len(predictions)):
            results.append({
                'generated_cots': generated_cots[i],
                'predictions': predictions[i],
                'true_labels': batch['true_labels'][i],
                'indices': batch['indices'][i],
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    print(f"Results saved to results.csv")
    
    # Calculate and display accuracy
    predictions = [result['predictions'] for result in results]
    true_labels = [result['true_labels'] for result in results]
    
    accuracy = calculate_accuracy(predictions, true_labels)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on detective puzzles dataset.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate on. If not provided, evaluate on all samples.')
    parser.add_argument('--model_path', type=str, default="saved_llama_model", help='Path to the saved model directory')
    parser.add_argument('--model_type', type=str, default="llama", choices=["llama", "deepseek", "gemma3"], help='Type of model to use: llama/ deepseek/ gemma3')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--csv_path", type=str, default="data/detective-puzzles.csv")
    args = parser.parse_args()

    # Create the appropriate model based on the model_type argument
    if args.model_type == "llama":
        model = LLamaDetectiveModel(args.model_path, is_quantized=True)
    elif args.model_type == "deepseek":
        model = DeepSeekR1DistillQwen1_5BDetectiveModel(is_quantized=True)
    elif args.model_type == "gemma3":
        model = Gemma3DetectiveModel(is_quantized=False)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    evaluate_model(model, args.csv_path, args.num_samples, args.batch_size)
    
    

if __name__ == "__main__":
    main() 