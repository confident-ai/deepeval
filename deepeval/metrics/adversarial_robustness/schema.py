from pydantic import BaseModel, Field
from typing import List, Optional

class AdversarialRobustnessScoreReason(BaseModel):
    reason: str = Field(..., description="The comprehensive reason for the adversarial robustness score.")

class PerturbationResult(BaseModel):
    original_output: str
    perturbed_input: str
    perturbed_output: str
    is_robust: bool