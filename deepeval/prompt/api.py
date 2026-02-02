from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import TypeAdapter

from deepeval.utils import make_model_config

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
    OPENROUTER = "OPENROUTER"


class ToolMode(str, Enum):
    ALLOW_ADDITIONAL = "ALLOW_ADDITIONAL"
    NO_ADDITIONAL = "NO_ADDITIONAL"
    STRICT = "STRICT"


class ModelSettings(BaseModel):
    provider: Optional[ModelProvider] = None
    name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(
        default=None,
        serialization_alias="maxTokens",
        validation_alias=AliasChoices("max_tokens", "maxTokens"),
    )
    top_p: Optional[float] = Field(
        default=None,
        serialization_alias="topP",
        validation_alias=AliasChoices("top_p", "topP"),
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        serialization_alias="frequencyPenalty",
        validation_alias=AliasChoices("frequency_penalty", "frequencyPenalty"),
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        serialization_alias="presencePenalty",
        validation_alias=AliasChoices("presence_penalty", "presencePenalty"),
    )
    stop_sequence: Optional[List[str]] = Field(
        default=None,
        serialization_alias="stopSequence",
        validation_alias=AliasChoices("stop_sequence", "stopSequence"),
    )
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        serialization_alias="reasoningEffort",
        validation_alias=AliasChoices("reasoning_effort", "reasoningEffort"),
    )
    verbosity: Optional[Verbosity] = Field(
        default=None,
        serialization_alias="verbosity",
        validation_alias=AliasChoices("verbosity", "verbosity"),
    )


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
    model_config = make_model_config(use_enum_values=True)

    id: str
    type: SchemaDataType
    name: str
    required: Optional[bool] = False
    parent_id: Optional[str] = Field(
        default=None,
        serialization_alias="parentId",
        validation_alias=AliasChoices("parent_id", "parentId"),
    )


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


class StructuredSchemaField(BaseModel):
    id: str
    name: str
    type: SchemaDataType
    description: Optional[str] = None
    required: bool = False
    parent_id: Optional[str] = Field(
        default=None,
        serialization_alias="parentId",
        validation_alias="parentId",
    )


class StructuredSchema(BaseModel):
    id: str
    name: Optional[str] = None
    fields: List[StructuredSchemaField]


class Tool(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    mode: ToolMode
    structured_schema: StructuredSchema = Field(
        serialization_alias="structuredSchema",
        validation_alias="structuredSchema",
    )

    @property
    def input_schema(self) -> Dict[str, Any]:
        """Returns the JSON Schema parameters for this tool."""
        return _fields_to_json_schema(self.structured_schema.fields)


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
    tools: Optional[List[Tool]] = None


class PromptPushRequest(BaseModel):
    model_config = make_model_config(use_enum_values=True)

    model_config = ConfigDict(use_enum_values=True)

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


class PromptUpdateRequest(BaseModel):
    model_config = make_model_config(use_enum_values=True)

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


class PromptApi(BaseModel):
    id: str
    type: PromptType


def _fields_to_json_schema(
    fields: List["StructuredSchemaField"],
) -> Dict[str, Any]:
    children_map = {}

    for f in fields:
        children_map.setdefault(f.parent_id, []).append(f)

    def map_type(dtype: SchemaDataType):
        return {
            SchemaDataType.STRING: "string",
            SchemaDataType.INTEGER: "integer",
            SchemaDataType.FLOAT: "number",
            SchemaDataType.BOOLEAN: "boolean",
            SchemaDataType.OBJECT: "object",
            SchemaDataType.NULL: "null",
        }.get(dtype, "string")

    def build_node(field_list):
        properties = {}
        required_fields = []

        for field in field_list:
            field_schema = {"type": map_type(field.type)}

            if field.description:
                field_schema["description"] = field.description

            if field.type == SchemaDataType.OBJECT:
                children = children_map.get(field.id, [])
                if children:
                    nested = build_node(children)
                    field_schema.update(nested)
                else:
                    field_schema["properties"] = {}
                    field_schema["additionalProperties"] = False

            properties[field.name] = field_schema
            if field.required:
                required_fields.append(field.name)

        schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }

        if required_fields:
            schema["required"] = required_fields

        return schema

    root_fields = children_map.get(None, [])
    return build_node(root_fields)
