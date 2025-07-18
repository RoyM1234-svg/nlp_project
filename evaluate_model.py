from typing import Optional
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score
from models.detective_model import DetectiveModel
from models.cot_models import *
from models.final_answer_models import *
from data_loaders.detective_data_loader import DetectiveDataLoader
from utils import extract_guilty_suspect


def calculate_accuracy(predictions, true_labels):
    """Calculate accuracy for the predictions."""
    pred_clean = [str(pred).strip().lower() for pred in predictions]
    true_clean = [str(true).strip().lower() for true in true_labels]
    
    accuracy = accuracy_score(true_clean, pred_clean)
    return accuracy


def evaluate_model(
        cot_model: DetectiveModel,
        final_answer_model: DetectiveModel,
        csv_path: str,
        num_samples: Optional[int],
        batch_size: int,
        model_type: str
        ):
    
    df = pd.read_csv(csv_path)
    if num_samples:
        df = df.sample(num_samples)
    data_loader = DetectiveDataLoader(df, batch_size=batch_size, shuffle=False)

    results = []

    for batch in tqdm(data_loader, desc="Generating batch"):
        mystery_texts = batch['mystery_texts']
        suspects_lists = batch['suspects_lists']
        true_labels = batch['true_labels']
        case_names = batch['case_names']
        indices = batch['indices']
        generated_cots = cot_model.generate_batch(mystery_texts, suspects_lists)
        raw_predictions = final_answer_model.generate_batch(mystery_texts, suspects_lists, generated_cots)
        predictions = [extract_guilty_suspect(suspects_list, prediction) for prediction, suspects_list in zip(raw_predictions, suspects_lists)]
        
        results.extend(zip(
            case_names,
            generated_cots,
            suspects_lists,
            predictions,
            true_labels,
            indices,
            raw_predictions,
            ))
    
    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['case_names', 'generated_cots', 'suspects_lists', 'predictions', 'true_labels', 'indices', 'raw_predictions'])
    results_df.to_csv(f"results_{model_type}.csv", index=False)
    print(f"Results saved to results_{model_type}.csv")
    
    predictions = [result[3] for result in results]  
    true_labels = [result[4] for result in results]  
    
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

    if args.model_type == "llama":
        cot_model = LLamaDetectiveModel(args.model_path, is_quantized=True)
        final_answer_model = LLamaFinalAnswerModel(args.model_path, is_quantized=True)
    elif args.model_type == "deepseek":
        cot_model = DeepSeekDetectiveModel(is_quantized=True)
        final_answer_model = DeepSeekFinalAnswerModel(is_quantized=True)
    elif args.model_type == "gemma3":
        cot_model = Gemma3DetectiveModel(is_quantized=False)
        final_answer_model = Gemma3FinalAnswerModel(is_quantized=False)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    evaluate_model(cot_model, final_answer_model, args.csv_path, args.num_samples, args.batch_size, args.model_type)
    
    

if __name__ == "__main__":
    main() 