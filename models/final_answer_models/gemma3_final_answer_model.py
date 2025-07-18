import re
from models.detective_model import DetectiveModel

class Gemma3FinalAnswerModel(DetectiveModel):
    """
    Final answer model using Google's Gemma 3 1B IT model from Hugging Face.
    """
    def __init__(
        self,
        is_quantized: bool = False,
        max_new_tokens: int = 100,
        do_sample: bool = False,
    ):
        super().__init__(
            model_path="google/gemma-3-1b-it",
            is_quantized=is_quantized,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
        """This model has a small context- up to 32768 tokens"""
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        user_prompt = f"""Mystery Story:
{mystery_text}

Suspects:
{suspects_list}

Solution:
{cot}

OUTPUT FORMAT: [Name only]

The guilty suspect is: """

        system_prompt = "You are an expert detective who identifies the guilty suspect by name. Output ONLY the name with no explanation or reasoning."

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
