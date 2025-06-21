from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.pipelines import pipeline
import torch
import re
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
import json


class DetectiveModel(ABC):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=2000,
                 temperature=0.7,
                 top_p=0.9,
                 stopping_criteria=None,
                ):
        self.model_path = model_path
        self.is_quantized = is_quantized
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stopping_criteria = stopping_criteria
        self.load_model()

    def load_model(self):        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,  # Standard choice for GPUs
            )
        
        if self.is_quantized:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def run_inference(self, mystery_text: str, suspects: list[str]) -> tuple[str, str]:
        prompt = self.create_prompt(mystery_text, suspects)
       
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature, 
            top_p=self.top_p, # only the top 90% of tokens are considered
            return_full_text=False, # deletes the [INST] tags
            stopping_criteria=self.stopping_criteria
        )
        full_response = outputs[0]['generated_text']
        predicted_suspect = self.extract_guilty_suspect(full_response)
        
        return full_response, predicted_suspect
    
    def run_inference_batch(self, mystery_texts: list[str], suspects_lists: list[list[str]]) -> list[tuple[str, str]]:
        pass
        
    # Private methods
    @abstractmethod
    def create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
        """Create a prompt for the model based on the mystery and suspects.
        
        Args:
            mystery_text: The mystery story text
            suspects: List of suspect names
            
        Returns:
            Formatted prompt string for the model
        """
        pass

    @staticmethod
    @abstractmethod
    def extract_guilty_suspect(full_response: str) -> str:
        """Extract the guilty suspect from the model response.
        
        Args:
            full_response: The full response from the model
            
        Returns:
            The guilty suspect name
        """
        pass
    

class DetectiveStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] <= self.prompt_length:
            return False
        # Check last 30 tokens for complete ==X== pattern
        text = self.tokenizer.decode(input_ids[0, -30:], skip_special_tokens=True)
        
        # Stop immediately when we see ==something==
        if re.search(r'==[^=]+==', text):
            return True
        return False
    