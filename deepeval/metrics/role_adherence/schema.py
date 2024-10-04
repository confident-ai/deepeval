from typing import List
from pydantic import BaseModel


class OutOfCharacterResponseIndicies(BaseModel):
    indicies: List[int]


class Reason(BaseModel):
    reason: str
