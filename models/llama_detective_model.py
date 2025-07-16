
import re
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from models.detective_model import DetectiveModel


class LLamaDetectiveModel(DetectiveModel):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=1000,
                 temperature=0.7,
                 top_p=0.9,
                 stopping_criteria=None,
                ):
        super().__init__(model_path, is_quantized, max_new_tokens, temperature,top_p,stopping_criteria)

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

class LLamaDetectiveStoppingCriteria(StoppingCriteria):
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