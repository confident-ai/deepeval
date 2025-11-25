from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class OptimizerDisplayConfig(BaseModel):
    """Display controls used by GEPA"""

    show_indicator: bool = True
    announce_ties: bool = Field(
        False, description="Print a one-line note when a tie is detected"
    )


class OptimizerPromptRolesConfig(BaseModel):
    """
    Controls how to map prompts and goldens to roles for list-style prompts
    and conversational test cases.
    """

    # Role used when appending the golden's primary input to a LIST prompt.
    # Defaults to "user".
    list_input_role: str = Field(
        "user",
        description=(
            "PromptMessage.role to use for the appended golden input when "
            "prompt.type is LIST."
        ),
    )

    # Role used to represent the human side when synthesizing conversational turns
    # from a ConversationalGolden, for example: from .scenario when no turns are provided
    conversational_user_role: str = Field(
        "user",
        description=(
            "Turn.role used for synthesized user side content when building "
            "ConversationalTestCase from ConversationalGolden."
        ),
    )

    # Role used for the model’s answer or actual output in conversational test cases.
    conversational_assistant_role: str = Field(
        "assistant",
        description=(
            "Turn.role used for the assistant / model output when building "
            "ConversationalTestCase from ConversationalGolden."
        ),
    )

    @field_validator(
        "conversational_user_role", "conversational_assistant_role"
    )
    @classmethod
    def _must_be_valid_turn_role(cls, role: str) -> str:
        # Turn.role currently only supports these literal values.
        allowed = {"user", "assistant"}
        if role not in allowed:
            raise ValueError(
                "OptimizerPromptRolesConfig.conversational_*_role must be "
                "'user' or 'assistant' to match Turn.role."
            )
        return role


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
