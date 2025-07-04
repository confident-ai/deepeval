from pydantic import BaseModel, Field
from typing import List


class ManipulationVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Reason(BaseModel):
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ManipulationVerdict] 