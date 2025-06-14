from huggingface_hub import snapshot_download
from pathlib import Path
import torch
import os
import pandas as pd

# Model setup and loading code
print("--- Setting up model path and downloading files ---")
mistral_models_path = Path("/content/mistral_models/7B-Instruct-v0.3")
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    allow_patterns=["*.json", "*.safetensors", "tokenizer.model.v3"],
    local_dir=mistral_models_path,
)

original_tokenizer_file = mistral_models_path / "tokenizer.model.v3"
expected_tokenizer_file = mistral_models_path / "tokenizer.model"
if original_tokenizer_file.exists() and not expected_tokenizer_file.exists():
    print("Renaming tokenizer file for transformers compatibility...")
    original_tokenizer_file.rename(expected_tokenizer_file)

print("\n--- Loading tokenizer and model onto the GPU ---")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(mistral_models_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    mistral_models_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("--- Model loading complete! ---")


def extract_final_answer(model_output):
    """
    Parses the model's text to find the final answer enclosed in ==*...*== markers.
    """
    try:
        start_marker = "==*"
        end_marker = "*=="
        start_index = model_output.rfind(start_marker)
        if start_index == -1: return "None (Marker not found)"
        end_index = model_output.find(end_marker, start_index)
        if end_index == -1: return "None (End marker not found)"
        culprit = model_output[start_index + len(start_marker):end_index]
        return culprit.strip()
    except Exception as e:
        return f"None (Error during parsing: {e})"


# Load Data, Solve, Evaluate, and Store Results
print("\n--- Loading mystery data ---")
csv_file_path = 'detective-puzzles.csv' 
df = pd.read_csv(csv_file_path)

results_list = []
total_cases = 0

for index, row in df.iterrows():
    if pd.isna(row['case_name']):
        continue
    total_cases += 1
    case_name = row['case_name']
    mystery_text = row['mystery_text']
    answer_options = row['answer_options']
    correct_answer = row['answer']
    
    print(f"\n==================== Solving Case: {case_name} ====================")
    
    prompt = f"""
    You are an expert detective. Read the following mystery and determine the most likely culprit from the list of options provided.

    **Mystery Story:**
    {mystery_text}

    **Answer Options:**
    {answer_options}

    Based on the evidence in the story, who is the culprit? First, explain your chain of thought. Then, to conclude your entire response, state the final answer formatted exactly like this: ==*culprit's name*==
    For example, if you believe the culprit is (a) John Smith, your response must end with: ==*(a) John Smith*==
    """

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    
    outputs = model.generate(inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    
    model_full_response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

    extracted_choice = extract_final_answer(model_full_response)

    print(f"\n--- Reasoning from Model ---\n{model_full_response.strip()}")
    print("\n--- Evaluation ---")
    print(f"Model's Extracted Choice: {extracted_choice}")
    print(f"Correct Answer: {correct_answer}")
    
    is_correct = extracted_choice.lower() == correct_answer.lower()
    
    if is_correct:
        print("Outcome: CORRECT! 🎉")
    else:
        print("Outcome: Incorrect.")

    results_list.append({
        'case_name': case_name,
        'model_full_response': model_full_response.strip(),
        'model_extracted_answer': extracted_choice,
        'correct_answer': correct_answer,
        'is_correct': is_correct
    })

# Save Results to CSV and Print Final Report

results_df = pd.DataFrame(results_list)

output_csv_path = 'evaluation_results.csv'
results_df.to_csv(output_csv_path, index=False)
print(f"\n\nEvaluation results saved to {output_csv_path}")


# Final Summary Report
print("\n\n==================== FINAL REPORT ====================")

correct_count = results_df['is_correct'].sum()
incorrect_count = total_cases - correct_count

if total_cases > 0:
    # --- Calculate Metrics ---
    # For this specific task (pick one right answer), Accuracy, Precision,
    # Recall, and F1-score all simplify to the same value.
    # True Positives (TP): The model got the answer right.
    # False Positives (FP): The model picked a name, but it was wrong.
    # False Negatives (FN): The model failed to pick the correct name.
    
    tp = correct_count
    fp = incorrect_count 
    # fn = incorrect_count 
    
    accuracy = tp / total_cases
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # # Recall = TP / (TP + FN)
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    # f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Total cases evaluated: {total_cases}")
    print(f"Correctly solved:      {correct_count}")
    print("-" * 20)
    print(f"Accuracy:  {accuracy:.2%}")
    # print(f"Precision: {precision:.2%}")
    # print(f"Recall:    {recall:.2%}")
    # print(f"F1 Score:  {f1_score:.2f}")

else:
    print("No cases were evaluated.")