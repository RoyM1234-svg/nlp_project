import re
from models.detective_model import DetectiveModel, DetectiveStoppingCriteria

class DeepSeekDetectiveModel(DetectiveModel):
    def create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        prompt = f"""<｜User｜>
Read the following mystery and determine who is guilty.

Mystery Story:
{mystery_text}

Suspects:
{suspects_list}

Instructions:
1. Analyze the evidence carefully
2. Explain your reasoning step by step
3. End with your final answer in this exact format: GUILTY: [name]
4. The guilty person MUST be one of the suspects listed above
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


class DeepSeekR1DistillQwen1_5BDetectiveModel(DetectiveModel):
    """
    Detective model using DeepSeek-R1-Distill-Qwen-1.5B from Hugging Face.
    """
    def __init__(
        self,
        is_quantized: bool = True,
        max_new_tokens: int = 2000,
        temperature: float = 1.0,
        top_p: float = 0.9,
        stopping_criteria: DetectiveStoppingCriteria = None,
    ):
        super().__init__(
            model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            is_quantized=is_quantized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stopping_criteria=stopping_criteria,
        )

#     def create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
#         """
#         Build the prompt instructing the model to output the exact culprit format.
#         """
#         answer_options = "\n".join(f"- {suspect}" for suspect in suspects)
#         prompt = f"""
# You are an expert detective. Read the following mystery and determine the most likely culprit from the options below.

# Mystery Story:
# {mystery_text}

# Answer Options:
# {answer_options}

# First, explain your chain of thought step by step. THEN, on the last line of your response, output **exactly** this format and nothing else:
# CULPRIT: [name from Answer Options]
# So you need to write CULPRIT in capial letters, followed by a colon and a space, and then the name of the culprit in square brackets.
# For example, if the culprit is John Smith, the final line must be:
# CULPRIT: [John Smith]
# """
#         return prompt

#     @staticmethod
#     def extract_guilty_suspect(full_response: str) -> str:
#         # Remove thinking tags if present
#         cleaned_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
        
#         # Extract guilty verdict
#         pattern = r'CULPRIT:\s*\[([^\]]+)\]|CULPRIT:\s*([^\n]+)'
#         matches = re.findall(pattern, cleaned_response)
#         if matches:
#             for match in matches:
#                 result = match[0] if match[0] else match[1]
#                 return result.strip()
#         return "Unknown"

    def create_prompt(self, mystery_text: str, suspects: list[str]) -> str:
        suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
        
        prompt = f"""<｜User｜>
Read the following mystery and determine who is guilty.

Mystery Story:
{mystery_text}

Suspects:
{suspects_list}

Instructions:
1. Analyze the evidence carefully
2. Explain your reasoning step by step
3. End with your final answer in this exact format: GUILTY: [name]
4. The guilty person MUST be one of the suspects listed above
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
