from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, conint
from typing import Optional
from deepeval.evaluate.configs import AsyncConfig


class DisplayConfig(BaseModel):
    show_indicator: bool = True
    announce_ties: bool = Field(
        False, description="Print a one-line note when a tie is detected"
    )


class MutationTargetType(Enum):
    RANDOM = "random"
    FIXED_INDEX = "fixed_index"


# default all messages
class MutationConfig(BaseModel):
    target_type: MutationTargetType = MutationTargetType.RANDOM
    # should be list
    target_role: Optional[str] = Field(
        default=None,
        description="If set, restricts candidates to messages with this role (case insensitive).",
    )
    target_index: conint(ge=0) = Field(
        default=0,
        description="0-based index used when target_type == FIXED_INDEX.",
    )
