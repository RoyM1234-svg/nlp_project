from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Enable GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model with optimizations for Colab Pro
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",  # Automatically handle device placement
    torch_dtype=torch.float16  # Use half precision for better memory efficiency
)

# Create pipeline with the model
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
for i, prompt in enumerate(inputs, 1):
    output = pipe(
        prompt,
        max_new_tokens=50,
        do_sample=False,
        temperature=0.7
    )[0]["generated_text"]
    print(f"--- Example {i} ---\nPrompt: {prompt}\nOutput: {output}\n")