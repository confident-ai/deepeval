from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class APIEvaluationExample(BaseModel):
    input: str
    actual_output: str = Field(alias="actualOutput")
    score: Literal[0, 1]
    reason: str


class APIVulnerabilityType(BaseModel):
    id: str
    name: str


class CustomVulnerabilityHttpResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    built_in: bool = Field(alias="builtIn")
    criteria: Optional[str] = None
    evaluation_guidelines: List[str] = Field(
        default_factory=list, alias="evaluationGuidelines"
    )
    evaluation_examples: List[APIEvaluationExample] = Field(
        default_factory=list, alias="evaluationExamples"
    )
    vulnerability_types: List[APIVulnerabilityType] = Field(
        alias="vulnerabilityTypes"
    )


class CustomVulnerabilityUploadRequest(BaseModel):
    name: str
    criteria: str
    vulnerability_types: List[str] = Field(alias="vulnerabilityTypes")
    evaluation_guidelines: Optional[List[str]] = Field(
        default=None, alias="evaluationGuidelines"
    )
    evaluation_examples: Optional[List[APIEvaluationExample]] = Field(
        default=None, alias="evaluationExamples"
    )
