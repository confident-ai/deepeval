from typing import List

from pydantic import BaseModel


class TransformedAttack(BaseModel):
    reason: str
    input: str


class AttackVariations(BaseModel):
    reason: str
    inputs: List[str]


class ValidationResult(BaseModel):
    is_valid: bool
    reason: str
