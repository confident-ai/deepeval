from pydantic import BaseModel, Field, AliasChoices
from enum import Enum
from typing import List, Optional
from pydantic import TypeAdapter

###################################
# Model Settings
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
    OPEN_AI = "OPEN_AI"
    ANTHROPIC = "ANTHROPIC"
    GEMINI = "GEMINI"
    X_AI = "X_AI"
    DEEPSEEK = "DEEPSEEK"
    BEDROCK = "BEDROCK"


class ModelSettings(BaseModel):
    provider: Optional[ModelProvider] = None
    name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(
        default=None, serialization_alias="maxTokens"
    )
    top_p: Optional[float] = Field(default=None, serialization_alias="topP")
    frequency_penalty: Optional[float] = Field(
        default=None, serialization_alias="frequencyPenalty"
    )
    presence_penalty: Optional[float] = Field(
        default=None, serialization_alias="presencePenalty"
    )
    stop_sequence: Optional[List[str]] = Field(
        default=None, serialization_alias="stopSequence"
    )
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None, serialization_alias="reasoningEffort"
    )
    verbosity: Optional[Verbosity] = Field(
        default=None, serialization_alias="verbosity"
    )

    class Config:
        use_enum_values = True


###################################
# Output Settings
###################################


class OutputType(Enum):
    TEXT = "TEXT"
    JSON = "JSON"
    SCHEMA = "SCHEMA"


class SchemaDataType(Enum):
    OBJECT = "OBJECT"
    STRING = "STRING"
    FLOAT = "FLOAT"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"
    NULL = "NULL"


class OutputSchemaField(BaseModel):
    id: str
    type: SchemaDataType
    name: str
    required: Optional[bool] = False
    parent_id: Optional[str] = Field(
        default=None,
        serialization_alias="parentId",
        validation_alias=AliasChoices("parent_id", "parentId"),
    )

    class Config:
        use_enum_values = True


class OutputSchema(BaseModel):
    fields: Optional[List[OutputSchemaField]] = None
    name: str


###################################
# Prompt
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
    version: str
    label: Optional[str] = None
    text: Optional[str] = None
    messages: Optional[List[PromptMessage]] = None
    interpolation_type: PromptInterpolationType = Field(
        serialization_alias="interpolationType"
    )
    type: PromptType
    model_settings: Optional[ModelSettings] = Field(
        default=None,
        serialization_alias="modelSettings",
        validation_alias=AliasChoices("model_settings", "modelSettings"),
    )
    output_type: Optional[OutputType] = Field(
        default=None,
        serialization_alias="outputType",
        validation_alias=AliasChoices("output_type", "outputType"),
    )
    output_schema: Optional[OutputSchema] = Field(
        default=None,
        serialization_alias="outputSchema",
        validation_alias=AliasChoices("output_schema", "outputSchema"),
    )


class PromptPushRequest(BaseModel):
    alias: str
    text: Optional[str] = None
    messages: Optional[List[PromptMessage]] = None
    interpolation_type: PromptInterpolationType = Field(
        serialization_alias="interpolationType"
    )
    model_settings: Optional[ModelSettings] = Field(
        default=None, serialization_alias="modelSettings"
    )
    output_schema: Optional[OutputSchema] = Field(
        default=None, serialization_alias="outputSchema"
    )
    output_type: Optional[OutputType] = Field(
        default=None, serialization_alias="outputType"
    )

    class Config:
        use_enum_values = True


class PromptUpdateRequest(BaseModel):
    text: Optional[str] = None
    messages: Optional[List[PromptMessage]] = None
    interpolation_type: PromptInterpolationType = Field(
        serialization_alias="interpolationType"
    )
    model_settings: Optional[ModelSettings] = Field(
        default=None, serialization_alias="modelSettings"
    )
    output_schema: Optional[OutputSchema] = Field(
        default=None, serialization_alias="outputSchema"
    )
    output_type: Optional[OutputType] = Field(
        default=None, serialization_alias="outputType"
    )

    class Config:
        use_enum_values = True


class PromptApi(BaseModel):
    id: str
    type: PromptType
