from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
        }
        
        if self.is_quantized:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )
        self.model.eval()

    @torch.no_grad()
    def generate_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[list[str], list[str]]:
        """Generate text for a batch of tokenized inputs."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_length = input_ids.shape[1]
        generated_ids = outputs[:, prompt_length:]
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        predicted_suspects = [self.extract_guilty_suspect(text) for text in generated_texts]
        
        return generated_texts, predicted_suspects
    
    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
        
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


    # --- Generate k samples ---
    @torch.no_grad()
    def generate_k_samples(self, mystery_text: str, suspects: list[str], k: int) -> list[tuple[str, str]]:
        """
        Generates k different outputs for a SINGLE prompt in one efficient call.

        Args:
            mystery_text: The mystery story text.
            suspects: List of suspect names.
            k: The number of different outputs to generate.

        Returns:
            A list of k (full_response, predicted_suspect) tuples.
        """
        prompt = self.create_prompt(mystery_text, suspects)
        
        # Tokenize the single prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate k sequences from the single prompt
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            # --- THE KEY PARAMETERS ---
            num_return_sequences=k,
        )

        # Decode all k outputs at once
        # We need to slice the output to remove the prompt part
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = outputs[:, prompt_length:]
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        # Extract suspects from each of the k generated texts
        results = []
        for text in generated_texts:
            predicted_suspect = self.extract_guilty_suspect(text)
            results.append((text, predicted_suspect))
            
        return results
    


