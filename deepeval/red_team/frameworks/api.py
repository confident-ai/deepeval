from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class APIEvaluationExample(BaseModel):
    input: str
    actual_output: str = Field(alias="actualOutput")
    score: Literal[0, 1]
    reason: str


class APIVulnerability(BaseModel):
    name: str
    types: List[str]
    criteria: Optional[str] = None
    evaluation_guidelines: Optional[List[str]] = Field(
        None, alias="evaluationGuidelines"
    )
    evaluation_examples: Optional[List[APIEvaluationExample]] = Field(
        None, alias="evaluationExamples"
    )


class APIAttackMethod(BaseModel):
    name: str
    multi_turn: bool = Field(alias="multiTurn")
    parameters: Optional[Dict[str, Any]] = None


class APIRiskCategory(BaseModel):
    name: str
    description: Optional[str] = None
    vulnerabilities: List[APIVulnerability]
    attack_methods: List[APIAttackMethod] = Field(alias="attackMethods")


class RedTeamingFrameworkHttpResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    risk_categories: List[APIRiskCategory] = Field(alias="riskCategories")
