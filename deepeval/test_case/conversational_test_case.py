from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
from pydantic import AnyUrl
from copy import deepcopy
from enum import Enum
from mcp.types import Tool, Resource, Prompt, CallToolResult, ReadResourceResult, GetPromptResult
from deepeval.test_case import ToolCall


class TurnParams(Enum):
    ROLE = "role"
    CONTENT = "content"
    SCENARIO = "scenario"
    EXPECTED_OUTCOME = "expected_outcome"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"


# @dataclass
# class MCPTool:
#     name: str
#     input_schema: Dict
#     output_schema: Dict
#     title: Optional[str] = None
#     description: Optional[str] = None


# @dataclass
# class MCPToolCall:
#     name: str
#     args: Dict
#     structured_content: Dict # can use the "result" property in this for ease of access instead of using content
#     is_error: bool
#     content: Optional[List] = None# Will have to implement content types later on if needed from the MCP types.py


# @dataclass
# class MCPPromptCall:
#     description: str
#     messages: List


# @dataclass
# class MCPResourceCall:
#     contents: List # Gotta use the .text / .blob    


# @dataclass
# class MCPResource:
#     name: str
#     mimeType: str
#     uri: AnyUrl
#     title: Optional[str] = None
#     description: Optional[str] = None


# @dataclass
# class MCPPrompt:
#     name: str
#     arguments: List
#     title: Optional[str] = None
#     description: Optional[str] = None


@dataclass
class MCPMetaData:
    server_name: str
    transport: Optional[Literal["stdio", "sse", "streamable-http"]] = None
    available_tools: Optional[List] = None
    available_resources: Optional[List] = None
    available_prompts: Optional[List] = None


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    user_id: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[ToolCall]] = None
    mcp_tools_called: Optional[List[Dict]] = None
    mcp_resources_called: Optional[List[Dict]] = None
    mcp_prompts_called: Optional[List[Dict]] = None
    additional_metadata: Optional[Dict] = None


@dataclass
class ConversationalTestCase:
    turns: List[Turn]
    chatbot_role: Optional[str] = None
    scenario: Optional[str] = None
    user_description: Optional[str] = None
    expected_outcome: Optional[str] = None
    name: Optional[str] = field(default=None)
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    mcp_data: Optional[List[MCPMetaData]] = None
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if len(self.turns) == 0:
            raise TypeError("'turns' must not be empty")

        copied_turns = []
        for turn in self.turns:
            if not isinstance(turn, Turn):
                raise TypeError("'turns' must be a list of `Turn`s")
            
            if turn.mcp_tools_called is not None:
                if not isinstance(turn.mcp_tools_called, List) or not all(
                    isinstance(tool_called, CallToolResult) for tool_called in turn.mcp_tools_called
                ):
                    raise TypeError("The 'tools_called' must be a list of 'CallToolResult' from mcp.types")
            
            if turn.mcp_resources_called is not None:
                if not isinstance(turn.mcp_resources_called, List) or not all(
                    isinstance(resource_called, ReadResourceResult) for resource_called in turn.mcp_resources_called
                ):
                    raise TypeError("The 'resources_called' must be a list of 'ReadResourceResult' from mcp.types")
                
            if turn.mcp_prompts_called is not None:
                if not isinstance(turn.mcp_prompts_called, List) or not all(
                    isinstance(prompt_called, GetPromptResult) for prompt_called in turn.mcp_prompts_called
                ):
                    raise TypeError("The 'prompts_called' must be a list of 'GetPromptResult' from mcp.types")
                
            copied_turns.append(deepcopy(turn))

        for mcp_data in self.mcp_data:

            if mcp_data.available_tools is not None:
                if not isinstance(mcp_data.available_tools, List) or not all(
                    isinstance(tool, Tool) for tool in mcp_data.available_tools
                ):
                    raise TypeError("'available_tools' must be a list of 'Tool' from mcp.types")
                
            if mcp_data.available_resources is not None:
                if not isinstance(mcp_data.available_resources, List) or not all(
                    isinstance(resource, Resource) for resource in mcp_data.available_resources
                ):
                    raise TypeError("'available_resources' must be a list of 'Resource' from mcp.types")
                
            if mcp_data.available_prompts is not None:
                if not isinstance(mcp_data.available_prompts, List) or not all(
                    isinstance(prompt, Prompt) for prompt in mcp_data.available_prompts
                ):
                    raise TypeError("'available_prompts' must be a list of 'Prompt' from mcp.types")

        self.turns = copied_turns
