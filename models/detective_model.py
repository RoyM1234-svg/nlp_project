from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.pipelines import pipeline
import torch
import re
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


class DetectiveModel(ABC):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=2000,
                 temperature=0.7,
                 use_stopping_criteria=False,
                ):
        self.model_path = model_path
        self.is_quantized = is_quantized
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_stopping_criteria = use_stopping_criteria
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
        if self.use_stopping_criteria:
            stopping_criteria = StoppingCriteriaList([DetectiveStoppingCriteria(self.tokenizer, len(prompt))])
        else:
            stopping_criteria = None
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature, 
            top_p=0.9, # only the top 90% of tokens are considered
            return_full_text=False, # deletes the [INST] tags
            stopping_criteria=stopping_criteria
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


class LLamaDetectiveModel(DetectiveModel):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=2000,
                 temperature=0.7,
                 use_stopping_criteria=False,
                ):
        super().__init__(model_path, is_quantized, max_new_tokens, temperature, use_stopping_criteria)

    def create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        system_prompt = "You are an expert detective who analyzes evidence and solves mysteries."
        
        user_prompt = f"""Read the following mystery and determine who is guilty.

        Mystery Story:
        {mystery_text}

        Suspects:
        {suspects_list}

        Instructions:
        1. Analyze the evidence carefully
        2. Explain your reasoning step by step
        3. End with your final answer in this exact format: GUILTY: [name]
        4. The guilty person MUST be one of the suspects listed above"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return prompt
    
    @staticmethod
    def extract_guilty_suspect(full_response: str) -> str:
        pattern = r'GUILTY:\s*\[([^\]]+)\]|GUILTY:\s*([^\n]+)'
        matches = re.findall(pattern, full_response)
        if matches:
            for match in matches:
                result = match[0] if match[0] else match[1]
                return result.strip()
        return "Unknown"
    

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