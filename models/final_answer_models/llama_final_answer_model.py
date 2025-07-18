from models.detective_model import DetectiveModel

class LLamaFinalAnswerModel(DetectiveModel):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=100,
                 do_sample=False
                 ):
        super().__init__(model_path, is_quantized, max_new_tokens, do_sample)

    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
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
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return prompt

