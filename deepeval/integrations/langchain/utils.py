from typing import Any, List, Dict
from langchain_core.outputs import ChatGeneration

def parse_prompts_to_messages(prompts: list[str]) -> List[Dict[str, str]]:
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

def convert_chat_generation_to_string(gen: ChatGeneration) -> str:
    return gen.message.pretty_repr()

def prepare_dict(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}