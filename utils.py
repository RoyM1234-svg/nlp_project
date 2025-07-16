"""
Utility functions for detective models.
"""

import re
from typing import Optional


def extract_guilty_suspect(response: str) -> str:
    """
    Extract the guilty suspect name from model response.
    
    Looks for patterns like:
    - GUILTY: [suspect name]
    - GUILTY: suspect name
    
    Args:
        response: The model's response text
        
    Returns:
        str: The extracted suspect name, or "Unknown" if not found
    """
    if not response:
        return "Unknown"
    
    cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    pattern = r'GUILTY:\s*\[?([^\]\n]+?)\]?(?:\s*$|\n)'
    matches = re.findall(pattern, cleaned_response, re.IGNORECASE)
    
    if matches:
        return matches[-1].strip()  # Take the last match in case of multiple
    
    return "Unknown" 