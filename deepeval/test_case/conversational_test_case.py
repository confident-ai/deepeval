from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
from pydantic import AnyUrl
from copy import deepcopy
from enum import Enum

from deepeval.test_case import ToolCall


class TurnParams(Enum):
    ROLE = "role"
    CONTENT = "content"
    SCENARIO = "scenario"
    EXPECTED_OUTCOME = "expected_outcome"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"


@dataclass
class MCPTool:
    name: str
    input_schema: Dict
    output_schema: Dict
    title: Optional[str] = None
    description: Optional[str] = None


@dataclass
class MCPToolCall:
    name: str
    args: Dict
    structured_content: Dict # can use the "result" property in this for ease of access instead of using content
    is_error: bool
    content: Optional[List] = None# Will have to implement content types later on if needed from the MCP types.py


@dataclass
class MCPPromptCall:
    description: str
    messages: List


@dataclass
class MCPResourceCall:
    contents: List # Gotta use the .text / .blob    


@dataclass
class MCPResource:
    name: str
    mimeType: str
    uri: AnyUrl
    title: Optional[str] = None
    description: Optional[str] = None


@dataclass
class MCPPrompt:
    name: str
    arguments: List
    title: Optional[str] = None
    description: Optional[str] = None


@dataclass
class MCPMetaData:
    server_name: str
    transport: Optional[Literal["stdio", "sse", "streamable-http"]] = None
    available_tools: Optional[List[MCPTool]] = None
    available_resources: Optional[List[MCPResource]] = None
    available_prompts: Optional[List[MCPPrompt]] = None


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    user_id: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[ToolCall]] = None
    additional_metadata: Optional[Dict] = None
    mcp_tools_called: Optional[List[MCPToolCall]] = None
    mcp_resources_called: Optional[List[MCPResourceCall]] = None
    mcp_prompts_called: Optional[List[MCPPromptCall]] = None


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
            copied_turns.append(deepcopy(turn))

        self.turns = copied_turns
