from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class OptimizerDisplayConfig(BaseModel):
    """Display controls used by GEPA"""

    show_indicator: bool = True
    announce_ties: bool = Field(
        False, description="Print a one-line note when a tie is detected"
    )


class PromptListMutationTargetType(Enum):
    FIRST = "first"
    RANDOM = "random"
    FIXED_INDEX = "fixed_index"


class PromptListMutationConfig(BaseModel):
    target_type: PromptListMutationTargetType = (
        PromptListMutationTargetType.FIRST
    )
    target_role: Optional[str] = Field(
        default=None,
        description="If set, restricts candidates to messages with this role (case insensitive).",
    )
    target_index: Optional[int] = Field(
        default=None,
        description="0-based index used when target_type == FIXED_INDEX.",
    )
