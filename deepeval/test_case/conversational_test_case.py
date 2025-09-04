from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    model_validator,
    AliasChoices,
)
from typing import List, Optional, Dict, Literal
from copy import deepcopy
from enum import Enum

from deepeval.test_case import ToolCall
from deepeval.test_case.mcp import (
    MCPServer,
    MCPPromptCall,
    MCPResourceCall,
    MCPToolCall,
    validate_mcp_servers,
)


class TurnParams(Enum):
    ROLE = "role"
    CONTENT = "content"
    SCENARIO = "scenario"
    EXPECTED_OUTCOME = "expected_outcome"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    MCP_TOOLS = "mcp_tools_called"
    MCP_RESOURCES = "mcp_resources_called"
    MCP_PROMPTS = "mcp_prompts_called"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    user_id: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("userId", "user_id")
    )
    retrieval_context: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("retrievalContext", "retrieval_context"),
    )
    tools_called: Optional[List[ToolCall]] = Field(
        default=None,
        validation_alias=AliasChoices("toolsCalled", "tools_called"),
    )
    mcp_tools_called: Optional[List[MCPToolCall]] = Field(default=None)
    mcp_resources_called: Optional[List[MCPResourceCall]] = Field(default=None)
    mcp_prompts_called: Optional[List[MCPPromptCall]] = Field(default=None)
    additional_metadata: Optional[Dict] = Field(
        default=None,
        serialization_alias="additionalMetadata",
        validation_alias=AliasChoices(
            "additionalMetadata", "additional_metadata"
        ),
    )
    _mcp_interaction: bool = PrivateAttr(default=False)

    def __repr__(self):
        attrs = [f"role={self.role!r}", f"content={self.content!r}"]
        if self.user_id is not None:
            attrs.append(f"user_id={self.user_id!r}")
        if self.retrieval_context is not None:
            attrs.append(f"retrieval_context={self.retrieval_context!r}")
        if self.tools_called is not None:
            attrs.append(f"tools_called={self.tools_called!r}")
        if self.mcp_tools_called is not None:
            attrs.append(f"mcp_tools_called={self.mcp_tools_called!r}")
        if self.mcp_resources_called is not None:
            attrs.append(f"mcp_resources_called={self.mcp_resources_called!r}")
        if self.mcp_prompts_called is not None:
            attrs.append(f"mcp_prompts_called={self.mcp_prompts_called!r}")
        if self.additional_metadata is not None:
            attrs.append(f"additional_metadata={self.additional_metadata!r}")
        return f"Turn({', '.join(attrs)})"

    @model_validator(mode="before")
    def validate_input(cls, data):
        mcp_tools_called = data.get("mcp_tools_called")
        mcp_prompts_called = data.get("mcp_prompts_called")
        mcp_resources_called = data.get("mcp_resources_called")

        if (
            mcp_tools_called is not None
            or mcp_prompts_called is not None
            or mcp_resources_called is not None
        ):
            from mcp.types import (
                CallToolResult,
                ReadResourceResult,
                GetPromptResult,
            )

            data["_mcp_interaction"] = True
            if mcp_tools_called is not None:
                if not isinstance(mcp_tools_called, list) or not all(
                    isinstance(tool_called, MCPToolCall)
                    and isinstance(tool_called.result, CallToolResult)
                    for tool_called in mcp_tools_called
                ):
                    raise TypeError(
                        "The 'tools_called' must be a list of 'MCPToolCall' with result of type 'CallToolResult' from mcp.types"
                    )

            if mcp_resources_called is not None:
                if not isinstance(mcp_resources_called, list) or not all(
                    isinstance(resource_called, MCPResourceCall)
                    and isinstance(resource_called.result, ReadResourceResult)
                    for resource_called in mcp_resources_called
                ):
                    raise TypeError(
                        "The 'resources_called' must be a list of 'MCPResourceCall' with result of type 'ReadResourceResult' from mcp.types"
                    )

            if mcp_prompts_called is not None:
                if not isinstance(mcp_prompts_called, list) or not all(
                    isinstance(prompt_called, MCPPromptCall)
                    and isinstance(prompt_called.result, GetPromptResult)
                    for prompt_called in mcp_prompts_called
                ):
                    raise TypeError(
                        "The 'prompts_called' must be a list of 'MCPPromptCall' with result of type 'GetPromptResult' from mcp.types"
                    )

        return data


class ConversationalTestCase(BaseModel):
    turns: List[Turn]
    scenario: Optional[str] = Field(default=None)
    context: Optional[List[str]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    user_description: Optional[str] = Field(
        default=None,
        serialization_alias="userDescription",
        validation_alias=AliasChoices("userDescription", "user_description"),
    )
    expected_outcome: Optional[str] = Field(
        default=None,
        serialization_alias="expectedOutcome",
        validation_alias=AliasChoices("expectedOutcome", "expected_outcome"),
    )
    chatbot_role: Optional[str] = Field(
        default=None,
        serialization_alias="chatbotRole",
        validation_alias=AliasChoices("chatbotRole", "chatbot_role"),
    )
    additional_metadata: Optional[Dict] = Field(
        default=None,
        serialization_alias="additionalMetadata",
        validation_alias=AliasChoices(
            "additionalMetadata", "additional_metadata"
        ),
    )
    comments: Optional[str] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    mcp_servers: Optional[List[MCPServer]] = Field(default=None)

    _dataset_rank: Optional[int] = PrivateAttr(default=None)
    _dataset_alias: Optional[str] = PrivateAttr(default=None)
    _dataset_id: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="before")
    def validate_input(cls, data):
        turns = data.get("turns")
        context = data.get("context")
        mcp_servers = data.get("mcp_servers")

        if len(turns) == 0:
            raise TypeError("'turns' must not be empty")

        # Ensure `context` is None or a list of strings
        if context is not None:
            if not isinstance(context, list) or not all(
                isinstance(item, str) for item in context
            ):
                raise TypeError("'context' must be None or a list of strings")

        if mcp_servers is not None:
            validate_mcp_servers(mcp_servers)

        copied_turns = []
        for turn in turns:
            if isinstance(turn, Turn):
                copied_turns.append(deepcopy(turn))
            elif isinstance(turn, dict):
                try:
                    copied_turns.append(Turn.model_validate(turn))
                except Exception as e:
                    raise TypeError(f"Invalid dict for Turn: {turn} ({e})")
            else:
                raise TypeError(
                    f"'turns' must be a list of Turn or dict, got {type(turn)}"
                )

        data["turns"] = copied_turns

        return data
