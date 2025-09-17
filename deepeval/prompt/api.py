from pydantic import BaseModel, Field, AliasChoices
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


class PromptVersion(BaseModel):
    id: str
    version: str
    commit_message: str = Field(
        serialization_alias="commitMessage",
        validation_alias=AliasChoices("commit_message", "commitMessage"),
    )


class PromptVersionsHttpResponse(BaseModel):
    text_versions: Optional[List[PromptVersion]] = Field(
        None,
        serialization_alias="textVersions",
        validation_alias=AliasChoices("text_versions", "textVersions"),
    )
    messages_versions: Optional[List[PromptVersion]] = Field(
        None,
        serialization_alias="messagesVersions",
        validation_alias=AliasChoices("messages_versions", "messagesVersions"),
    )


class PromptHttpResponse(BaseModel):
    id: str
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
