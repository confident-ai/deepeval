from typing import List, Dict, Union
from pydantic import BaseModel, Field


class CoherenceVerdict(BaseModel):
    index: int
    verdict: str
    reason: str = Field(default=None)
