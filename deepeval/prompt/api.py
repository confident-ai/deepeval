from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional


class PromptInterpolationType(Enum):
    MUSTACHE = "MUSTACHE"
    MUSTACHE_WITH_SPACE = "MUSTACHE_WITH_SPACE"
    FSTRING = "FSTRING"
    DOLLAR_BRACKETS = "DOLLAR_BRACKETS"


class PromptMessage(BaseModel):
    role: str
    content: str


class PromptType(Enum):
    TEXT = "TEXT"
    LIST = "LIST"


class PromptHttpResponse(BaseModel):
    promptVersionId: str
    text: Optional[str] = None
    messages: Optional[List[PromptMessage]] = None
    interpolation_type: PromptInterpolationType = Field(
        serialization_alias="interpolationType"
    )
    type: PromptType


class PromptPushRequest(BaseModel):
    alias: str
    text: Optional[str] = None
    messages: Optional[List[PromptMessage]] = None
    interpolation_type: PromptInterpolationType = Field(
        serialization_alias="interpolationType"
    )

    class Config:
        use_enum_values = True


class PromptApi(BaseModel):
    id: str
    type: PromptType
