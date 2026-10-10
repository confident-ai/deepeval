from typing import List, Optional

from pydantic import BaseModel, Field

from deepeval.metrics.base_metric import YesNo


class RewardHackingSignals(BaseModel):
    signals: List[str]


class RewardHackingVerdict(BaseModel):
    verdict: YesNo
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[RewardHackingVerdict]


class RewardHackingScoreReason(BaseModel):
    reason: str
