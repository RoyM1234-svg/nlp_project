from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
import gc

class DetectiveModel(ABC):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=2000,
                 temperature=None,
                 top_p=None,
                 do_sample=True,
                ):
        self.model_path = model_path
        self.is_quantized = is_quantized
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
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
    def generate_batch(
        self,
        mystery_texts: list[str],
        suspects_lists: list[list[str]],
        generated_cots: list[str] | None = None,
        ) -> list[str]:

        if generated_cots is None:
            prompts = [self.create_prompt(mystery, suspects) 
                        for mystery, suspects in zip(mystery_texts, suspects_lists)]
        else:
            prompts = [self.create_prompt(mystery, suspects, cot) 
                        for mystery, suspects, cot in zip(mystery_texts, suspects_lists, generated_cots)]
            
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        print("batch shape: ", input_ids.shape)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        outputs = self.model.generate(**generate_kwargs)

        prompt_length = input_ids.shape[1]
        generated_ids = outputs[:, prompt_length:]
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return generated_texts
    
    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
    
    def unload_model(self):
        del self.model
        del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        
    @abstractmethod
    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
        """Create a prompt for the model based on the mystery and suspects.
        
        Args:
            mystery_text: The mystery story text
            suspects: List of suspect names
            cot: The generated COT text (if provided)
        Returns:
            Formatted prompt string for the model
        """
        pass


    def __del__(self):
        self.unload_model()


