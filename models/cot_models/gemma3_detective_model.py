import re
from models.detective_model import DetectiveModel



class Gemma3DetectiveModel(DetectiveModel):
    """
    Detective model using Google's Gemma 3 1B IT model from Hugging Face.
    """
    def __init__(
        self,
        is_quantized: bool = False,
        max_new_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        super().__init__(
            model_path="google/gemma-3-1b-it",
            is_quantized=is_quantized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
        """This model has a small context- up to 32768 tokens"""
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        # Base instruction (exact from research, adapted for names)
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
            add_generation_prompt=True,
            tokenize=False
        )

        return prompt