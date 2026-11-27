from pydantic import BaseModel


class EnhancedOverride(BaseModel):
    authority_strategy: str
    input: str


class IsOverrideValid(BaseModel):
    is_valid_override: bool
