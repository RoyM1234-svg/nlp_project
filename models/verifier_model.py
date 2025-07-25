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
    

    
    