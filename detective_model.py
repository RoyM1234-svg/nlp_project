from transformers import AutoModelForCausalLM
from transformers.pipelines import pipeline
import torch
import re

class DetectiveModel:
    def __init__(self, model_path, is_quantized=True, max_new_tokens=2000, temperature=0.7):
        self.model_path = model_path
        self.is_quantized = is_quantized
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_model()

    def load_model(self):
        if self.is_quantized:
            # Load your saved model WITH 8-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,  # Your saved model directory
                load_in_8bit=True,  # Apply quantization during loading
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # Load without quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def run_inference(self, mystery_text: str, suspects: list[str]) -> tuple[str, str]:
        prompt = self._create_prompt(mystery_text, suspects)
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature, #randomness
            top_p=0.9, # only the top 90% of tokens are considered
            return_full_text=False # deletes the [INST] tags
        )
        full_response = outputs[0]['generated_text']
        predicted_suspect = self._extract_guilty_suspect(full_response)
        
        return full_response, predicted_suspect
    
    def run_inference_batch(self, mystery_texts: list[str], suspects_lists: list[list[str]]) -> list[tuple[str, str]]:


    # Private methods
    def _create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        prompt = f"""
        You are an expert detective. Read the following mystery and determine the most likely guilty suspect from the list of options provided.

        *Mystery Story:*
        {mystery_text}

        *Suspect List:*
        {suspects_list}

        Based on the evidence in the story, who is the guilty suspect? First, explain your chain of thought. Then, to conclude your entire response, state the final answer formatted exactly like this: ==guilty suspect's name==
        For example, if you believe the guilty suspect is John Smith, your response must end with: ==John Smith==
        """
        
        return prompt
    
    
    @staticmethod
    def _extract_guilty_suspect(full_response: str) -> str:
        # Look for the pattern ==suspect name== at the end of the response
        pattern = r'==([^=]+)=='
        matches = re.findall(pattern, full_response)
        if matches:
            return matches[-1].strip()
        else:
            print("No suspect found in expected format '==suspect name==' in the model response")
            return "Unknown"
        
        