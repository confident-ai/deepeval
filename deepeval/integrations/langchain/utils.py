from typing import List, Dict

def parse_prompts_to_messages(prompts: list[str]) -> List[Dict[str, str]]:
    """
    Parse a list of strings in format '<role>: <message>' to a list of dictionaries.
    If a string doesn't match the format, default role is 'Human'.
    
    Args:
        prompts: List of strings in format '<role>: <message>'
        
    Returns:
        List of dictionaries with role as key and message as value
    """
    messages = []
    for prompt in prompts:
        for message in prompt.split("\n"):
            if ': ' in message:
                role, message = message.split(': ', 1)
                messages.append({
                    "role": role.strip(),
                    "content": message.strip()
                })
            else:
                messages.append({
                    "role": "Human",
                    "content": message.strip()
                })
    
    return messages