from models.detective_model import DetectiveModel

class LLamaFinalAnswerModel(DetectiveModel):
    def __init__(self,
                 model_path,
                 is_quantized=True,
                 max_new_tokens=100,
                 do_sample=False
                 ):
        super().__init__(
            model_path=model_path,
            is_quantized=is_quantized,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

    def create_prompt(self, mystery_text: str, suspects: list[str], cot: str | None = None) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])

        user_prompt = f"""
You are a detective assistant. Output ONLY the guilty suspect's name. 
Do NOT explain. Do NOT echo the reasoning. Do NOT add punctuation or extra tokens.

Mystery Story:
The victim was poisoned with a rare herb. Alice is on CCTV across town at the time of death.
Carol is severely allergic to that herb and cannot handle it safely.
Bob was recorded purchasing that exact herb yesterday and had access to the victim’s tea.

Suspects:
- Alice
- Bob
- Carol

Chain-of-thought (INPUT ONLY — DO NOT OUTPUT):
Alice has an airtight alibi. Carol’s allergy makes handling the herb improbable.
Bob bought the herb and had opportunity. Therefore, Bob is guilty.

Valid outputs: Alice | Bob | Carol

Final answer (name only):
Bob
        Mystery Story:
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

