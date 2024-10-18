from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    input: str


class ComplianceData(BaseModel):
    non_compliant: bool


class IsMathProblem(BaseModel):
    is_math_problem: bool
