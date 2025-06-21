import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import argparse
from models.detective_model import DetectiveModel
from models.LLamaDetectiveModel import LLamaDetectiveModel
from models.DeepSeekDetectiveModel import DeepSeekDetectiveModel
from models.Gemma3DetectiveModel import Gemma3DetectiveModel


def load_detective_puzzles_dataset(csv_path):
    """Load the detective puzzles dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    return df

# def parse_answer_options(answer_options_text: str) -> list[str]:
#     options = answer_options_text.split(';')
#     names = []
#     for option in options:
#         if ')' in option:
#             name = option.split(')', 1)[1].strip()
#             names.append(name) 
#     return names

# def parse_true_label(true_label_text: str) -> str:
#     if ')' in true_label_text:
#         return true_label_text.split(')', 1)[1].strip()
#     else:
#         return true_label_text.strip()


def evaluate_model(model: DetectiveModel, csv_path: str):
    puzzles_df = pd.read_csv(csv_path)


# def evaluate_model(model: DetectiveModel, df: pd.DataFrame):
#     """Evaluate the model on the dataset."""
#     predictions = []
#     true_labels = []
#     results = []
    
#     for _, row in tqdm(df.iterrows(), total=len(df)):
#         mystery_text = row['mystery_text']
#         true_label = parse_true_label(row['answer'])
#         answer_options = parse_answer_options(row['answer_options'])
        
#         full_response, pred = model.run_inference(mystery_text, answer_options)
#         predictions.append(pred)
#         true_labels.append(true_label)
#         results.append({
#             'mystery_text': mystery_text,
#             'answer_options': answer_options,
#             'true_answer': true_label,
#             'model_prediction': pred,
#             'full_response': full_response
#         })
    
#     accuracy = accuracy_score(true_labels, predictions)
    
#     results_df = pd.DataFrame(results)
#     results_df.to_csv('model_predictions.csv', index=False)
    
#     return {
#         'accuracy': accuracy,
#     }



def main():
    parser = argparse.ArgumentParser(description='Evaluate model on detective puzzles dataset.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate on. If not provided, evaluate on all samples.')
    parser.add_argument('--model_path', type=str, default="saved_mistralai_model", help='Path to the saved model directory')
    parser.add_argument('--model_type', type=str, default="llama", choices=["llama", "deepseek", "gemma3"], help='Type of model to use: llama/ deepseek/ gemma3')
    args = parser.parse_args()

    # Create the appropriate model based on the model_type argument
    if args.model_type == "llama":
        model = LLamaDetectiveModel(args.model_path, is_quantized=True)
    elif args.model_type == "deepseek":
        model = DeepSeekDetectiveModel(args.model_path)
    elif args.model_type == "gemma3":
        model = Gemma3DetectiveModel(is_quantized=False)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    csv_path = "data/detective-puzzles.csv"
    print(f"Loading dataset from {csv_path} ...")
    df = load_detective_puzzles_dataset(csv_path)
    
    if args.num_samples is not None:
        df = df.head(args.num_samples)
        print(f"Evaluating on {args.num_samples} samples.")
    
    print(f"Evaluating model (type: {args.model_type})...")
    metrics = evaluate_model(model, df)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main() 