import re
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from models.detective_model import DetectiveModel


class LLamaFinalAnswerModel(DetectiveModel):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=1,
                 temperature=0.1,
                 top_p=0.5
                 ):
        super().__init__(model_path, is_quantized, max_new_tokens, temperature, top_p)

    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])

        system_prompt = "You are an expert detective who gives a concise final verdict."

        user_prompt = f"""You will receive:
        - A mystery story
        - A list of suspects
        - A chain-of-thought analysis explaining the reasoning

        Based on these, decide who is guilty in ONE WORD only (the suspect's name), without any additional explanation.

        Mystery Story:
        {mystery_text}

        Suspects:
        {suspects_list}

        Chain-of-Thought:
        {cot}

        Output:
        Respond with ONLY the guilty suspect's name, and nothing else. It must exactly match one of the suspects listed above."""

        return self.create_prompt_template(system_prompt, user_prompt, self.tokenizer)

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