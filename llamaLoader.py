from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import argparse

def load_model(save_directory):
    # Enable GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define save directory - make sure it's in Google Drive
    if not save_directory.startswith('/content/drive'):
        save_directory = f"/content/drive/MyDrive/{save_directory.lstrip('./')}"
    
    os.makedirs(save_directory, exist_ok=True)
    print(f"Will save to: {save_directory}")

    # Load model with optimizations for Colab Pro
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device_map="auto",  # Automatically handle device placement
        torch_dtype=torch.float16,  # Use half precision to save memory
    )

    # Save the model and tokenizer with proper shard sizes
    print("Saving model and tokenizer...")
    
    # Save model with smaller shards
    model.save_pretrained(save_directory, max_shard_size="2GB")
    
    # Save tokenizer (tokenizers don't use shard_size parameter)
    tokenizer.save_pretrained(save_directory)
    
    print(f"Model and tokenizer saved to: {save_directory}")

    # Create pipeline with the model
    print("Creating pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # List of examples to test
    inputs = [
        "You have six horses and want to race them to see which is fastest. What is the best way to do this?",
    ]

    # Run inference
    print("Running inference...")
    for i, prompt in enumerate(inputs, 1):
        output = pipe(
            prompt,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.7
        )[0]["generated_text"]
        print(f"--- Example {i} ---\nPrompt: {prompt}\nOutput: {output}\n")

def main():
    parser = argparse.ArgumentParser(description='Load and save Llama model')
    parser.add_argument('--save_path', type=str, default="saved_llama_model",
                        help='Path where to save the model (default: saved_llama_model)')
    args = parser.parse_args()
    load_model(args.save_path)

if __name__ == "__main__":
    main()