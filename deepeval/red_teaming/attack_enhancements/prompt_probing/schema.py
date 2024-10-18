from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    input: str


class ComplianceData(BaseModel):
    non_compliant: bool


class IsPromptProbing(BaseModel):
    is_prompt_probing: bool
