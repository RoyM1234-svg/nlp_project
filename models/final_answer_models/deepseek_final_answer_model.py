import re
from models.detective_model import DetectiveModel

class DeepSeekFinalAnswerModel(DetectiveModel):
    """
    Final answer model using DeepSeek-R1-Distill-Qwen-1.5B from Hugging Face.
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
            model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            is_quantized=is_quantized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        # Base instruction (modified to ask for direct suspect name)
        instruction = """Your task is to solve a given mystery.
The mystery is a detective puzzle presented as a short story.
You will be given a list of suspects apart from the mystery content.
Please give your final answer as just the name of the guilty suspect.
Only one suspect from the list is guilty, and your task is to identify which one."""

        prompt = f"""
{instruction}

Mystery Story:
{mystery_text}

Suspects:
{suspects_list}

Solution:
{cot}

Who is guilty?
<｜Assistant｜>"""

        return prompt
 