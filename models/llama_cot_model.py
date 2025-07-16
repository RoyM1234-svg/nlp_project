from models.detective_model import DetectiveModel

class LLamaCotModel(DetectiveModel):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=2000,
                 temperature=0.7,
                 top_p=0.9,
                 ):
        super().__init__(model_path, is_quantized, max_new_tokens, temperature, top_p)

    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])

        system_prompt = "You are an expert detective who analyzes evidence and solves mysteries."

        user_prompt = f"""Read the following mystery and determine who is guilty.

        Mystery Story:
        {mystery_text}

        Suspects:
        {suspects_list}

        Instructions:
        1. Analyze the evidence carefully
        2. Explain your reasoning step by step
        3. End with your final answer in this exact format: GUILTY: [name]
        4. The guilty person MUST be one of the suspects listed above"""

        return self.create_prompt_template(system_prompt, user_prompt, self.tokenizer)