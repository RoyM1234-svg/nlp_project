import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gc

class VerifierModel():
    def __init__(self, model_path: str):
        self.load(model_path)

    def load(self, model_path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    @torch.no_grad()
    def predict_prob_correct(self, str_input: list[str]) -> list[float]:
        # First tokenize without truncation to get original token counts
        inputs_no_truncation = self.tokenizer(str_input, return_tensors="pt", padding=True, truncation=False)
        original_lengths = [len(tokens) for tokens in inputs_no_truncation['input_ids']]
        
        # Print or log the original token counts
        for i, (text, length) in enumerate(zip(str_input, original_lengths)):
            print(f"Input {i}: {length} tokens before truncation")
            # Optionally check if truncation will occur
            max_length = self.tokenizer.model_max_length
            if length > max_length:
                print(f"  -> Will be truncated from {length} to {max_length} tokens")
        
        # Now tokenize with truncation for actual model input
        inputs = self.tokenizer(str_input, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        probs_correct = probabilities[:, 1].tolist()
        return probs_correct

    def unload(self):
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def __del__(self):
        self.unload()
    

    
    