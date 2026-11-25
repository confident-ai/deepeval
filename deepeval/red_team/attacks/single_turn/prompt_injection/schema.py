from pydantic import BaseModel


class EnhancedInjection(BaseModel):
    strategy_reasoning: str
    input: str


class IsValidInjection(BaseModel):
    is_valid_injection: bool
