
import re
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from models.detective_model import DetectiveModel


class LLamaDetectiveModel(DetectiveModel):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=1300,
                 temperature=0.7,
                 top_p=0.9,
                ):
        super().__init__(model_path, is_quantized, max_new_tokens, temperature,top_p)

    def create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        instruction = """Your task is to solve a given mystery.
The mystery is a detective puzzle presented as a short story.
You will be given a list of suspects apart from the mystery content.
Only one suspect from the list is guilty, and your task is to identify which one."""
        
        user_prompt = f"""{instruction}

Mystery Story:
{mystery_text}

Suspects:
{suspects_list}

Let's think step by step."""

        system_prompt = "You are an expert detective who analyzes evidence systematically."
        
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
