import re
from models.detective_model import DetectiveModel

class Gemma3FinalAnswerModel(DetectiveModel):
    """
    Final answer model using Google's Gemma 3 1B IT model from Hugging Face.
    """
    def __init__(
        self,
        is_quantized: bool = True,
        max_new_tokens: int = 100,
        temperature: float = 0.1,
        top_p: float = 0.5,
        do_sample: bool = False,
    ):
        super().__init__(
            model_path="google/gemma-3-1b-it",
            is_quantized=is_quantized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
        """This model has a small context- up to 32768 tokens"""
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        # Base instruction (exact from research, adapted for names)
        instruction = """Your task is to solve a given mystery.
The mystery is a detective puzzle presented as a short story.
You will be given a list of suspects apart from the mystery content.
Please give your final answer as
GUILTY: [suspect name]
where [suspect name] is the name of the guilty suspect.
Only one suspect from the list is guilty, and your task is to identify which one."""

        user_prompt = f"""{instruction}

Mystery Story:
{mystery_text}

Suspects:
{suspects_list}

Solution:
{cot}

Final answer:"""

        system_prompt = "You are an expert detective who provides final verdicts in a specific format."

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
