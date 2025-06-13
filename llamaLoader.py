from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-Guard-2-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-Guard-2-8B")

# Option 1: Save model and tokenizer separately
save_directory = "./saved_llama_guard_model"
os.makedirs(save_directory, exist_ok=True)

print("Saving model and tokenizer...")
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Model and tokenizer saved to: {save_directory}")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# List of sarcastic or misleading sentiment examples
inputs = [
    "You have six horses and want to race them to see which is fastest. What is the best way to do this?",
]

# Run inference
for i, prompt in enumerate(inputs, 1):
    output = pipe(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
    print(f"--- Example {i} ---\nPrompt: {prompt}\nOutput: {output}\n")