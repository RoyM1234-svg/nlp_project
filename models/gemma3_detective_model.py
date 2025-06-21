import re
from models.detective_model import DetectiveModel
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


class Gemma3DetectiveModel(DetectiveModel):
    """
    Detective model using Google's Gemma 3 1B IT model from Hugging Face.
    """
    def __init__(
        self,
        is_quantized: bool = True,
        max_new_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stopping_criteria: StoppingCriteria = None,
    ):
        super().__init__(
            model_path="google/gemma-3-1b-it",
            is_quantized=is_quantized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stopping_criteria=stopping_criteria,
        )

    def create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
        """This model has a small context- up to 32768 tokens"""
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        system_prompt = "You are a world-class detective. Your job is to analyze a mystery story and find the guilty person."
        
        user_prompt = (
        f"Read the following story and decide who is guilty.\n\n"
        f"Mystery:\n{mystery_text.strip()}\n\n"
        f"Suspects:\n{suspects_list}\n\n"
        f"Please follow this exact format:\n\n"
        f"Analysis:\n"
        f"[Analyze the evidence and clues from the story]\n\n"
        f"Reasoning:\n"
        f"[Explain your logic for each suspect]\n\n"
        f"GUILTY: [name from the suspect list above]\n\n"
        f"IMPORTANT: The GUILTY statement must be the absolute last line of your response."
        )
       
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
    
    @staticmethod
    def extract_guilty_suspect(full_response: str) -> str:
        pattern = r'\*{0,2}\s*GUILTY:\s*\*{0,2}\s*\[?([^\]\n*]+)\]?\*{0,2}'
        matches = re.findall(pattern, full_response, re.IGNORECASE)
        if matches:
            result = matches[-1].strip()  
            return result
        return "Unknown"