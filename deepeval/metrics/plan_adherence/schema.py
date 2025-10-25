from pydantic import BaseModel
from typing import List, Dict, Literal


class AgentPlan(BaseModel):
    plan: List[str]


class PlanAdherenceScore(BaseModel):
    score: float
    reason: str
