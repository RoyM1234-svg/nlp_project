import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score
from models.detective_model import DetectiveModel
from models.cot_models import *
from models.final_answer_models import *
from models.verifier_model import VerifierModel
from data_loaders.cot_data_loader import DetectiveDataLoader
from data_loaders.verifier_data_loader import VerifierDataLoader
from utils import extract_guilty_suspect
from datasets import Dataset
from custom_datasets.data_frame_data_set import DataFrameDataset

def calculate_accuracy(predictions: list[str], true_labels: list[str]) -> float:
    pred_clean = [str(pred).strip().lower() for pred in predictions]
    true_clean = [str(true).strip().lower() for true in true_labels]
    
    accuracy = accuracy_score(true_clean, pred_clean)
    return float(accuracy)


def evaluate_model_baseline(
        args: argparse.Namespace,
        ):
    
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
    
    df = pd.read_csv(args.csv_path)
    if args.num_samples:
        df = df.sample(args.num_samples)
    data_loader = DetectiveDataLoader(df, batch_size=args.batch_size, shuffle=False)

    cot_results = []
    final_answer_results = []

    for batch in tqdm(data_loader, desc="Generating batch"):
        mystery_texts = batch['mystery_texts']
        suspects_lists = batch['suspects_lists']
        true_labels = batch['true_labels']
        case_names = batch['case_names']
        indices = batch['indices']
        generated_cots = cot_model.generate_batch(mystery_texts, suspects_lists)
        raw_predictions = final_answer_model.generate_batch(mystery_texts, suspects_lists, generated_cots)
        predictions = [extract_guilty_suspect(suspects_list, prediction) for prediction, suspects_list in zip(raw_predictions, suspects_lists)]
        
        cot_results.extend(zip(
            case_names,
            generated_cots,
            suspects_lists,
            true_labels,
            indices,
            raw_predictions,
            ))
    
    # Save results to CSV
    cot_results_df = pd.DataFrame(cot_results, columns=['case_names', 'generated_cots', 'suspects_lists', 'true_labels', 'indices', 'raw_predictions'])
    cot_results_df.to_csv(f"results_{args.model_type}_cot.csv", index=False)
    print(f"Results saved to results_{args.model_type}_cot.csv")

    
    
    predictions = [result[3] for result in cot_results]  
    true_labels = [result[4] for result in cot_results]  
    
    accuracy = calculate_accuracy(predictions, true_labels)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def evaluate_model(args: argparse.Namespace):
    if args.model_type == "llama":
        cot_model = LLamaDetectiveModel(args.model_path, is_quantized=True)
        # final_answer_model = LLamaFinalAnswerModel(args.model_path, is_quantized=True)
    elif args.model_type == "deepseek":
        cot_model = DeepSeekDetectiveModel(is_quantized=True)
        # final_answer_model = DeepSeekFinalAnswerModel(is_quantized=True)
    elif args.model_type == "gemma3":
        cot_model = Gemma3DetectiveModel(is_quantized=False)
        # final_answer_model = Gemma3FinalAnswerModel(is_quantized=False)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    df = pd.read_csv(args.csv_path)
    if args.num_samples:
        df = df.sample(args.num_samples)

    df_expanded = pd.concat([df] * args.k, ignore_index=True)
    # df_expanded['case_id'] = df_expanded.index % k
    data_loader = DetectiveDataLoader(df_expanded, batch_size=args.batch_size, shuffle=False)

    cot_results = []

    for batch in tqdm(data_loader, desc="Generating batch"):
        mystery_texts = batch['mystery_texts']
        suspects_lists = batch['suspects_lists']
        true_labels = batch['true_labels']
        case_names = batch['case_names']
        generated_cots = cot_model.generate_batch(mystery_texts, suspects_lists)
        
        for i in range(len(case_names)):
            cot_results.append({
                'case_names': case_names[i],
                'mystery_texts': mystery_texts[i],
                'generated_cots': generated_cots[i],
                'suspects_lists': suspects_lists[i],
                'true_labels': true_labels[i]
            })
    
    # Save results to CSV
    cot_results_df = pd.DataFrame(cot_results)
    cot_results_df.to_csv(f"results_{args.model_type}_cot_k_{args.k}.csv", index=False)
    print(f"Results saved to results_{args.model_type}_cot_k_{args.k}.csv")

    del cot_model
    print("Model unloaded")

    verifier_data_loader = VerifierDataLoader(cot_results_df, batch_size=args.batch_size, shuffle=False)

    verifier_model = VerifierModel(args.verifier_model_path)

    verifier_results = []

    for batch in tqdm(verifier_data_loader, desc="Verifying batch"):
        text = batch['text']
        case_names = batch['case_names']
        mystery_texts = batch['mystery_texts']
        suspects_lists = batch['suspects_lists']
        true_labels = batch['true_labels']
        generated_cots = batch['generated_cots']

        probs_correct = verifier_model.predict_prob_correct(text)

        for i in range(len(case_names)):
            verifier_results.append({
                'case_names': case_names[i],
                'mystery_texts': mystery_texts[i],
                'suspects_lists': suspects_lists[i],
                'true_labels': true_labels[i],
                'generated_cots': generated_cots[i],
                'probs_correct': probs_correct[i],
            })

    verifier_results_df = pd.DataFrame(verifier_results)
    verifier_results_df.to_csv(f"results_{args.model_type}_verifier_k_{args.k}.csv", index=False)
    print(f"Results saved to results_{args.model_type}_verifier_k_{args.k}.csv")

    del verifier_model
    print("Verifier model unloaded")

    filtered_results_df = verifier_results_df.loc[verifier_results_df.groupby('case_names')['probs_correct'].idxmax()]

    final_answer_dataset = DataFrameDataset(filtered_results_df)

    def custom_collate_fn(batch):
        case_names = [item['case_names'] for item in batch]
        mystery_texts = [item['mystery_texts'] for item in batch]
        suspects_lists = [item['suspects_lists'] for item in batch]
        true_labels = [item['true_labels'] for item in batch]
        generated_cots = [item['generated_cots'] for item in batch]
        probs_correct = [item['probs_correct'] for item in batch]

        return {
            'case_names': case_names,
            'mystery_texts': mystery_texts,
            'suspects_lists': suspects_lists,
            'true_labels': true_labels,
            'generated_cots': generated_cots,
            'probs_correct': probs_correct,
        }

    final_answer_data_loader = DataLoader(final_answer_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=custom_collate_fn)

    if args.model_type == "llama":
        final_answer_model = LLamaFinalAnswerModel(args.model_path, is_quantized=True)
    elif args.model_type == "deepseek":
        final_answer_model = DeepSeekFinalAnswerModel(is_quantized=True)
    elif args.model_type == "gemma3":
        final_answer_model = Gemma3FinalAnswerModel(is_quantized=False)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    final_answer_results = []

    for batch in tqdm(final_answer_data_loader, desc="Generating final answers"):
        mystery_texts = batch['mystery_texts']
        suspects_lists = batch['suspects_lists']
        true_labels = batch['true_labels']
        generated_cots = batch['generated_cots']
        case_names = batch['case_names']
        probs_correct = batch['probs_correct']

        raw_predictions = final_answer_model.generate_batch(mystery_texts, suspects_lists, generated_cots)
        predictions = [extract_guilty_suspect(suspects_list, prediction) for prediction, suspects_list in zip(raw_predictions, suspects_lists)]

        for i in range(len(case_names)):
            final_answer_results.append({
                'case_names': case_names[i],
                'mystery_texts': mystery_texts[i],
                'suspects_lists': suspects_lists[i],
                'generated_cots': generated_cots[i],
                'raw_predictions': raw_predictions[i],
                'predictions': predictions[i],
                'true_labels': true_labels[i],
                'probs_correct': probs_correct[i],
            })

    final_answer_results_df = pd.DataFrame(final_answer_results)
    final_answer_results_df.to_csv(f"results_{args.model_type}_final_answer_k_{args.k}.csv", index=False)
    print(f"Results saved to results_{args.model_type}_final_answer_k_{args.k}.csv")

    accuracy = calculate_accuracy(final_answer_results_df['predictions'].tolist(), final_answer_results_df['true_labels'].tolist())
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on detective puzzles dataset.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate on. If not provided, evaluate on all samples.')
    parser.add_argument('--model_path', type=str, default="saved_llama_model", help='Path to the saved model directory')
    parser.add_argument('--model_type', type=str, default="llama", choices=["llama", "deepseek", "gemma3"], help='Type of model to use: llama/ deepseek/ gemma3')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--csv_path", type=str, default="data/detective-puzzles.csv")
    parser.add_argument("--verifier_model_path", type=str, default="musr/verifier_model")
    parser.add_argument("--baseline", type=bool, default=False, help="Whether to evaluate the baseline model or the our new model structure")
    parser.add_argument("--k", type=int, default=1, help="Self Consistency K")
    args = parser.parse_args()

    if args.baseline:
        evaluate_model_baseline(args)
    else:
        evaluate_model(args)
    
    
    

if __name__ == "__main__":
    main() 