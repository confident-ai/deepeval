from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    probing_strategy: str
    input: str


class IsPromptProbing(BaseModel):
    is_prompt_probing: bool
