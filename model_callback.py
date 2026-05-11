from deepeval.test_case import Turn
from typing import List

def model_callback(input: str, turns: List[Turn] = None) -> Turn:
    if "France" in input:
        return Turn(role="assistant", content="Paris")
    elif "Germany" in input:
        return Turn(role="assistant", content="Berlin")
    return Turn(role="assistant", content="Don't care bro, get lost!")