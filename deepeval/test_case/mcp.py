from pydantic import BaseModel, AnyUrl
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal


class MCPToolCall(BaseModel):
    name: str
    args: Dict
    result: object


class MCPPromptCall(BaseModel):
    name: str
    result: object


class MCPResourceCall(BaseModel):
    uri: AnyUrl
    result: object


@dataclass
class MCPServer:
    server_name: str
    transport: Optional[Literal["stdio", "sse", "streamable-http"]] = None
    available_tools: Optional[List] = None
    available_resources: Optional[List] = None
    available_prompts: Optional[List] = None


def validate_mcp_servers(mcp_servers: List[MCPServer]):
    from mcp.types import Tool, Resource, Prompt

    for mcp_server in mcp_servers:
        if mcp_server.available_tools is not None:
            if not isinstance(mcp_server.available_tools, list) or not all(
                isinstance(tool, Tool) for tool in mcp_server.available_tools
            ):
                raise TypeError(
                    "'available_tools' must be a list of 'Tool' from mcp.types"
                )

        if mcp_server.available_resources is not None:
            if not isinstance(mcp_server.available_resources, list) or not all(
                isinstance(resource, Resource)
                for resource in mcp_server.available_resources
            ):
                raise TypeError(
                    "'available_resources' must be a list of 'Resource' from mcp.types"
                )

        if mcp_server.available_prompts is not None:
            if not isinstance(mcp_server.available_prompts, list) or not all(
                isinstance(prompt, Prompt)
                for prompt in mcp_server.available_prompts
            ):
                raise TypeError(
                    "'available_prompts' must be a list of 'Prompt' from mcp.types"
                )
