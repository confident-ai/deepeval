from typing import List, Optional, Literal
from pydantic import BaseModel, Field

PerturbationType = Literal["semantic", "orthographic"]


class Perturbation(BaseModel):
    perturbed_input: str
    perturbation_type: PerturbationType


class Perturbations(BaseModel):
    perturbations: List[Perturbation]


# The AdversarialRobustnessMetric is inspired by the RoMA framework:
# "Towards Robust LLMs: an Adversarial Robustness Measurement Framework"
# (https://arxiv.org/abs/2504.17723).
class RobustnessVerdict(BaseModel):
    verdict: Literal["yes", "no"]
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[RobustnessVerdict]


class AdversarialRobustnessScoreReason(BaseModel):
    reason: str
