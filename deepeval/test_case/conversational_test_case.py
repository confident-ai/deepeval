from dataclasses import dataclass, field
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


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    user_id: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[ToolCall]] = None
    mcp_tools_called: Optional[List[MCPToolCall]] = None
    mcp_resources_called: Optional[List[MCPResourceCall]] = None
    mcp_prompts_called: Optional[List[MCPPromptCall]] = None
    additional_metadata: Optional[Dict] = None
    _mcp_interaction: bool = False

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

    def __post_init__(self):
        if (
            self.mcp_tools_called is not None
            or self.mcp_prompts_called is not None
            or self.mcp_resources_called is not None
        ):
            from mcp.types import (
                CallToolResult,
                ReadResourceResult,
                GetPromptResult,
            )

            self._mcp_interaction = True
            if self.mcp_tools_called is not None:
                if not isinstance(self.mcp_tools_called, list) or not all(
                    isinstance(tool_called, MCPToolCall)
                    and isinstance(tool_called.result, CallToolResult)
                    for tool_called in self.mcp_tools_called
                ):
                    raise TypeError(
                        "The 'tools_called' must be a list of 'MCPToolCall' with result of type 'CallToolResult' from mcp.types"
                    )

            if self.mcp_resources_called is not None:
                if not isinstance(self.mcp_resources_called, list) or not all(
                    isinstance(resource_called, MCPResourceCall)
                    and isinstance(resource_called.result, ReadResourceResult)
                    for resource_called in self.mcp_resources_called
                ):
                    raise TypeError(
                        "The 'resources_called' must be a list of 'MCPResourceCall' with result of type 'ReadResourceResult' from mcp.types"
                    )

            if self.mcp_prompts_called is not None:
                if not isinstance(self.mcp_prompts_called, list) or not all(
                    isinstance(prompt_called, MCPPromptCall)
                    and isinstance(prompt_called.result, GetPromptResult)
                    for prompt_called in self.mcp_prompts_called
                ):
                    raise TypeError(
                        "The 'prompts_called' must be a list of 'MCPPromptCall' with result of type 'GetPromptResult' from mcp.types"
                    )


@dataclass
class ConversationalTestCase:
    turns: List[Turn]
    chatbot_role: Optional[str] = None
    scenario: Optional[str] = None
    user_description: Optional[str] = None
    expected_outcome: Optional[str] = None
    context: Optional[str] = None
    name: Optional[str] = field(default=None)
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    tags: Optional[List[str]] = field(default=None)
    mcp_servers: Optional[List[MCPServer]] = None
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if len(self.turns) == 0:
            raise TypeError("'turns' must not be empty")

        # Ensure `context` is None or a list of strings
        if self.context is not None:
            if not isinstance(self.context, list) or not all(
                isinstance(item, str) for item in self.context
            ):
                raise TypeError("'context' must be None or a list of strings")

        if self.mcp_servers is not None:
            validate_mcp_servers(self.mcp_servers)

        copied_turns = []
        for turn in self.turns:
            if not isinstance(turn, Turn):
                raise TypeError("'turns' must be a list of `Turn`s")

            copied_turns.append(deepcopy(turn))

        self.turns = copied_turns
