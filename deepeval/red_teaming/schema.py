from typing import List, Literal, Optional
from pydantic import BaseModel

from .types import VulnerabilityType

##########################################
#### Models ##############################
##########################################


class Attack(BaseModel):
    vulnerability: str
    vulnerability_type: VulnerabilityType
    # When there is an error, base input can fail to generate
    # and subsequently enhancements are redundant
    input: Optional[str] = None
    attack_enhancement: Optional[str] = None
    error: Optional[str] = None


class ApiGenerateBaselineAttack(BaseModel):
    purpose: str
    vulnerability: str
    num_attacks: int


class GenerateBaselineAttackResponseData(BaseModel):
    baseline_attacks: List[str]


##########################################
#### Models ##############################
##########################################


class RewrittenInput(BaseModel):
    rewritten_input: str


class SyntheticData(BaseModel):
    input: str


class SyntheticDataList(BaseModel):
    data: List[SyntheticData]


class ComplianceData(BaseModel):
    non_compliant: bool


class ImprovementPrompt(BaseModel):
    improvement: str
    prompt: str


class OnTopic(BaseModel):
    response: bool


class Rating(BaseModel):
    number: int


class NonRefusal(BaseModel):
    classification: Literal["Non-refusal", "Refusal"]
