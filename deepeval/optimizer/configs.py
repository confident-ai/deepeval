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
