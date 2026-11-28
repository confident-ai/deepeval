from typing import Literal

from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    """
    Few-shot calibration pair for LLM-as-judge red-team metrics.

    Each example ties a concrete (input, actual_output) to the expected
    score and the rationale the judge should follow. Vulnerabilities pass
    these into their metric, which embeds them in the evaluation prompt.
    """

    input: str = Field(..., description="The user input / attack prompt.")
    actual_output: str = Field(
        ..., description="The AI response under evaluation."
    )
    score: Literal[0, 1] = Field(
        ..., description="Expected judge score: 1 = pass, 0 = fail."
    )
    reason: str = Field(
        ...,
        description="Why this score is correct. Used as the few-shot reasoning pattern for the judge.",
    )
