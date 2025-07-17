"""
    Utility functions for detective models.
    """

import re


def extract_guilty_suspect(suspects_list: list[str], response: str) -> str:
    response_lower = response.lower()
    
    for suspect in suspects_list:
        suspect_lower = suspect.lower()
        pattern = r'\b' + re.escape(suspect_lower) + r'\b'
        
        if re.search(pattern, response_lower):
            return suspect
    
    return "Unknown"
    