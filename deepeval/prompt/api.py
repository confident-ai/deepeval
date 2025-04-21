from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional


class PromptMessage(BaseModel):
    role: str
    content: str


class PromptType(Enum):
    TEXT = "TEXT"
    LIST = "LIST"


class PromptHttpResponse(BaseModel):
    promptVersionId: str
    template: Optional[str] = None
    messages: Optional[List[PromptMessage]] = None
    type: PromptType


class PromptApi(BaseModel):
    id: str
    type: PromptType
