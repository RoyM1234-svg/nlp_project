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
        do_sample: bool = False,
    ):
        super().__init__(
            model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            is_quantized=is_quantized,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])

        prompt = f"""You are a detective assistant. Output ONLY the name of the guilty suspect with no explanation or reasoning.

Mystery Story:
{mystery_text}

Suspects:
{suspects_list}

Solution:
{cot}

OUTPUT FORMAT: [Name only]

The guilty suspect is: """

        return prompt
 