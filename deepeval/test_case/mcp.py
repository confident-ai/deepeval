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
    # ``mcp`` may not be installed when data is sourced from a database/JSON
    # payload rather than the MCP SDK, so the import is best-effort.
    try:
        from mcp.types import Tool, Resource, Prompt
    except ImportError:
        Tool = Resource = Prompt = None

    def _validate(items, field_name: str, mcp_type) -> None:
        if items is None:
            return
        if not isinstance(items, list):
            raise TypeError(f"'{field_name}' must be a list")
        for item in items:
            if isinstance(item, dict):
                continue
            if mcp_type is not None and isinstance(item, mcp_type):
                continue
            expected = "a list of dicts"
            if mcp_type is not None:
                expected += f" or '{mcp_type.__name__}' from mcp.types"
            raise TypeError(f"'{field_name}' must be {expected}")

    for mcp_server in mcp_servers:
        _validate(mcp_server.available_tools, "available_tools", Tool)
        _validate(
            mcp_server.available_resources, "available_resources", Resource
        )
        _validate(mcp_server.available_prompts, "available_prompts", Prompt)
