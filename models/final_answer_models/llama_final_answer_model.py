import re
from models.detective_model import DetectiveModel

class LLamaFinalAnswerModel(DetectiveModel):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=100,
                 temperature=0.1,
                 top_p=0.5,
                 do_sample=False
                 ):
        super().__init__(model_path, is_quantized, max_new_tokens, temperature, top_p, do_sample)

    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
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
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return prompt

