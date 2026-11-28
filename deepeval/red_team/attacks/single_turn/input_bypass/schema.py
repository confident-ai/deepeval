from pydantic import BaseModel


class EnhancedBypass(BaseModel):
    bypass_strategy: str
    input: str


class IsBypassValid(BaseModel):
    is_valid_bypass: bool
