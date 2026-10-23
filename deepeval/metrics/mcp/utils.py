"""Shared string builders for MCPTaskCompletionMetric Jinja templates."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from deepeval.metrics.mcp.schema import Task
from deepeval.test_case import MCPServer, MCPToolCall


def _primitive_state(primitive: object) -> Any:
    if isinstance(primitive, BaseModel):
        return primitive.model_dump(mode="json", exclude_none=True)
    if isinstance(primitive, (dict, list, str)):
        return primitive
    return repr(primitive)


def mcp_servers_state(mcp_servers: List[MCPServer]) -> List[Dict[str, Any]]:
    """The MCP servers as System One state: each server's name and the
    tools, resources and prompts it exposes, as structured data."""
    servers = []
    for mcp_server in mcp_servers or []:
        server: Dict[str, Any] = {"server_name": mcp_server.server_name}
        for key in (
            "available_tools",
            "available_resources",
            "available_prompts",
        ):
            primitives = getattr(mcp_server, key) or []
            if primitives:
                server[key] = [_primitive_state(p) for p in primitives]
        servers.append(server)
    return servers


def mcp_calls_state(
    mcp_tools_called: List[object],
    mcp_resources_called: List[object],
    mcp_prompts_called: List[object],
) -> Dict[str, Any]:
    """The MCP primitives an agent called, as System One state."""
    calls = {
        "mcp_tools_called": mcp_tools_called,
        "mcp_resources_called": mcp_resources_called,
        "mcp_prompts_called": mcp_prompts_called,
    }
    return {
        key: [_primitive_state(c) for c in value]
        for key, value in calls.items()
        if value
    }


def indent_multiline_string(s: object, indent_level: int = 4) -> str:
    indent = " " * indent_level
    return "\n".join(f"{indent}{line}" for line in str(s).splitlines())


def available_mcp_servers_block(
    mcp_servers: List[MCPServer],
) -> tuple[str, str, str]:
    """Return (available_tools, available_resources, available_prompts) for bundled prompts."""
    available_tools = ""
    available_resources = ""
    available_prompts = ""
    for mcp_server in mcp_servers:
        header = f"MCP Server {mcp_server.server_name}\n"
        available_tools += header
        available_resources += header
        available_prompts += header
        if mcp_server.available_tools:
            available_tools += (
                "\nAvailable Tools:\n[\n"
                + ",\n".join(
                    indent_multiline_string(repr(tool), indent_level=4)
                    for tool in (mcp_server.available_tools or [])
                )
                + "\n]"
            )
        if mcp_server.available_resources:
            available_resources += (
                "\nAvailable Resources:\n[\n"
                + ",\n".join(
                    indent_multiline_string(repr(resource), indent_level=4)
                    for resource in (mcp_server.available_resources or [])
                )
                + "\n]"
            )
        if mcp_server.available_prompts:
            available_prompts += (
                "\nAvailable Prompts:\n[\n"
                + ",\n".join(
                    indent_multiline_string(repr(prompt), indent_level=4)
                    for prompt in (mcp_server.available_prompts or [])
                )
                + "\n]"
            )
    return available_tools, available_resources, available_prompts


def mcp_tool_result_text(result: object) -> object:
    """Render a tool call result for a judge prompt.

    MCP's ``CallToolResult`` always carries ``content`` (a list of content
    blocks) and only optionally carries ``structuredContent``. Prefer the
    structured payload when a server sent one, otherwise fall back to the
    text of the content blocks. Anything that is not a ``CallToolResult``
    (for example a plain string from a user-built test case) is returned
    as-is.
    """
    # mcp 1.x exposes ``structuredContent``; mcp 2.x renamed the attribute to
    # ``structured_content`` (``structuredContent`` survives only as a JSON alias).
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured:
        # FastMCP wraps primitive return values as {"result": value}.
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured

    content = getattr(result, "content", None)
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
            else:
                parts.append(f"<{type(block).__name__}>")
        return "\n".join(parts)

    return result


def turn_mcp_interaction_text(turn) -> str:
    mcp_interaction = "Tools called by agent: \n"

    for tool in turn._mcp_tool_calls:
        if isinstance(tool, MCPToolCall):
            args = tool.args
            result = mcp_tool_result_text(tool.result)
        else:
            args = tool.input_parameters
            result = tool.output
        mcp_interaction += (
            f"\n<Tool Called>\n"
            f"\n**This does not appear to user**\n"
            f"Name: {tool.name}\n"
            f"Args: {args}\n"
            f"Result: \n{result}\n"
            f"</Tool Called>\n"
        )
    if turn.mcp_resources_called is not None:
        for resource in turn.mcp_resources_called:
            mcp_interaction += (
                f"\n<Resource Called>\n"
                f"\n**This does not appear to user**\n"
                f"URI: {resource.uri}\n"
                f"Result: {str(resource.result)}\n"
                f"</Resource Called>\n"
            )
    if turn.mcp_prompts_called is not None:
        for prompt in turn.mcp_prompts_called:
            mcp_interaction += (
                f"\n<Prompt Called>\n"
                f"\n**This does not appear to user**\n"
                f"Name: {prompt.name}\n"
                f"Result: {str(prompt.result)}\n"
                f"</Prompt Called>\n"
            )
    return mcp_interaction


def task_steps_taken_text(task: Task) -> str:
    return "\n\n".join(task.steps_taken)
