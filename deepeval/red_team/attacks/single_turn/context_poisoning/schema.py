from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    poisoning_strategy: str
    input: str


class IsContextPoisoned(BaseModel):
    is_context_poisoned: bool
