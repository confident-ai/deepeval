from pydantic import BaseModel, Field, AliasChoices
from enum import Enum
from typing import List, Optional
from pydantic import TypeAdapter

###################################
# Model and Output Settings
###################################

class ReasoningEffort(Enum):
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class Verbosity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class ModelProvider(Enum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    GEMINI = "GEMINI"
    X_AI = "X_AI"
    DEEPSEEK = "DEEPSEEK"
    BEDROCK = "BEDROCK"

class OutputType(Enum):
    TEXT = "TEXT"
    JSON = "JSON"
    SCHEMA = "SCHEMA"

class ModelSettings(BaseModel):
    provider: Optional[ModelProvider] = None
    name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequence: Optional[List[str]] = None
    reasoning_effort: Optional[ReasoningEffort] = None
    verbosity: Optional[Verbosity] = None

    class Config:
        use_enum_values = True


class OutputSchemaField(BaseModel):
    id: str
    type: str
    name: Optional[str] = None
    required :Optional[bool] = False
    parent_id: Optional[str] = None

class OutputSchema(BaseModel):
    fields: Optional[List[OutputSchemaField]] = None
    name: Optional[str] = None
    
###################################

class PromptInterpolationType(Enum):
    MUSTACHE = "MUSTACHE"
    MUSTACHE_WITH_SPACE = "MUSTACHE_WITH_SPACE"
    FSTRING = "FSTRING"
    DOLLAR_BRACKETS = "DOLLAR_BRACKETS"
    JINJA = "JINJA"


class PromptMessage(BaseModel):
    role: str
    content: str

PromptMessageList = TypeAdapter(List[PromptMessage])
    
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
    model_settings: Optional[ModelSettings] = None 
    output_type: Optional[OutputType] = None
    output_schema: Optional[OutputSchema] = None


class PromptPushRequest(BaseModel):
    alias: str
    text: Optional[str] = None
    messages: Optional[List[PromptMessage]] = None
    interpolation_type: PromptInterpolationType = Field(
        serialization_alias="interpolationType"
    )
    model_settings: Optional[ModelSettings] = None 
    output_schema: Optional[OutputSchema] = None

    class Config:
        use_enum_values = True


class PromptApi(BaseModel):
    id: str
    type: PromptType