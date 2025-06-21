
import argparse
from models.detective_model import LLamaDetectiveModel
from data_loaders.detective_data_loader import DetectiveDataLoader
import pandas as pd
from tqdm import tqdm

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="saved_llama_model")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    # Load the saved model
    model_path = args.model_path
    model = LLamaDetectiveModel(model_path, is_quantized=True, use_stopping_criteria=False)

    print(f"Model path: {model_path}")
    print(f"Tokenizer class: {type(model.tokenizer)}")
    print(f"Pad token: '{model.tokenizer.pad_token}'")
    print(f"Pad token ID: {model.tokenizer.pad_token_id}")
    print(f"EOS token: '{model.tokenizer.eos_token}'")
    print(f"EOS token ID: {model.tokenizer.eos_token_id}")
    print(f"All special tokens: {model.tokenizer.special_tokens_map}")

    df = pd.read_csv("data/detective-puzzles.csv")
    df = df.sample(args.num_samples)
    data_loader = DetectiveDataLoader(model, df, batch_size=args.batch_size, shuffle=False)

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
    
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print(f"Results saved to results.csv")