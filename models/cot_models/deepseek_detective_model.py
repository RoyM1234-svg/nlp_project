import re
from models.detective_model import DetectiveModel

class DeepSeekDetectiveModel(DetectiveModel):
    """
    Detective model using DeepSeek-R1-Distill-Qwen-1.5B from Hugging Face.
    """
    def __init__(
        self,
        is_quantized: bool = True,
        max_new_tokens: int = 1500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        super().__init__(
            model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            is_quantized=is_quantized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    def create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        instruction = """Your task is to solve a given mystery.
The mystery is a detective puzzle presented as a short story.
You will be given a list of suspects apart from the mystery content.
Only one suspect from the list is guilty, and your task is to identify which one."""
        
        prompt = f"""<｜User｜>
{instruction}

Mystery Story:
{mystery_text}

Suspects:
{suspects_list}

Let's think step by step.
<｜Assistant｜>"""
        
        return prompt
    
    @staticmethod
    def extract_guilty_suspect(full_response: str) -> str:
        # Remove thinking tags if present
        cleaned_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
        
        # Extract guilty verdict
        pattern = r'GUILTY:\s*\[([^\]]+)\]|GUILTY:\s*([^\n]+)'
        matches = re.findall(pattern, cleaned_response)
        if matches:
            for match in matches:
                result = match[0] if match[0] else match[1]
                return result.strip()
        return "Unknown"
