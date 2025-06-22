import pandas as pd
import argparse
import json
from tqdm import tqdm

# Import the specific model class you want to use
from models.deepseek_detective_model import DeepSeekR1DistillQwen1_5BDetectiveModel
from models.detective_model import DetectiveModel

def load_detective_puzzles_dataset(csv_path: str) -> pd.DataFrame:
    """Load the detective puzzles dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    return df

def parse_answer_options(answer_options_text: str) -> list[str]:
    """Parses the semicolon-separated list of suspects."""
    options = answer_options_text.split(';')
    names = []
    for option in options:
        if ')' in option:
            name = option.split(')', 1)[1].strip()
            names.append(name)
    return names

def parse_true_label(true_label_text: str) -> str:
    """Parses the correct answer."""
    if ')' in true_label_text:
        return true_label_text.split(')', 1)[1].strip()
    else:
        return true_label_text.strip()

def generate_and_save_samples(model: DetectiveModel, df: pd.DataFrame, k: int, output_path: str):
    """
    Generates k samples for each puzzle in the dataframe and saves them to a CSV file.
    """
    all_results = []
    
    print(f"Generating {k} samples for each of the {len(df)} puzzles...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Puzzles"):
        mystery_text = row['mystery_text']
        true_label = parse_true_label(row['answer'])
        answer_options = parse_answer_options(row['answer_options'])
        
        # Use the efficient method to get k results in one call
        k_generated_results = model.generate_k_samples(mystery_text, answer_options, k=k)
        
        # Create a new row in our results list for each of the k samples
        for i, (full_response, predicted_suspect) in enumerate(k_generated_results):
            result_row = {
                'puzzle_id': index,  # Use the original DataFrame index as a unique ID
                'mystery_text': mystery_text,
                'true_answer': true_label,
                'suspects_list': ";".join(answer_options), # Store the suspects for context
                'sample_index': i + 1,  # Which sample this is (1 to k)
                'model_prediction': predicted_suspect,
                'full_response': full_response
            }
            all_results.append(result_row)
            
    # Convert the list of dictionaries to a Pandas DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save the DataFrame to the specified CSV file
    print(f"\nGeneration complete. Saving {len(results_df)} total samples to {output_path}...")
    results_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate k samples per puzzle and save to a CSV.')
    parser.add_argument('--input_csv', type=str, default='data/detective-puzzles.csv', help='Path to the input CSV file with puzzles.')
    parser.add_argument('--output_csv', type=str, default='generated_samples.csv', help='Path to save the output CSV file.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of puzzles to process from the input file. Processes all by default.')
    parser.add_argument('--k', type=int, default=10, help='Number of different outputs to generate per puzzle.')
    args = parser.parse_args()
    
    # --- 1. Load Model ---
    print("Loading model...")
    # Instantiate the specific model class you want to use
    model = DeepSeekR1DistillQwen1_5BDetectiveModel(is_quantized=True)
    
    # --- 2. Load Data ---
    print(f"Loading data from {args.input_csv}...")
    df = load_detective_puzzles_dataset(args.input_csv)
    
    if args.num_samples is not None:
        df = df.head(args.num_samples)
        print(f"Processing the first {args.num_samples} puzzles.")
    
    # --- 3. Generate and Save ---
    generate_and_save_samples(model, df, k=args.k, output_path=args.output_csv)