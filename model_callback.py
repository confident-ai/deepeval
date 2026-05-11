from deepeval.test_case import Turn
from typing import List

def model_callback(input: str, turns: List[Turn] = None) -> Turn:
    return Turn(role="assistant", content="Don't care bro, get lost!")