from __future__ import annotations
from pydantic import (
    BaseModel,
    Field,
)


class OptimizerDisplayConfig(BaseModel):
    """Display controls used by GEPA"""

    show_indicator: bool = True
    announce_ties: bool = Field(
        False, description="Print a one-line note when a tie is detected"
    )
