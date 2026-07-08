"""Shared string builders for MCPTaskCompletionMetric Jinja templates."""

from __future__ import annotations

from typing import List

from deepeval.metrics.mcp.schema import Task
from deepeval.test_case import MCPServer, MCPToolCall, ToolCall


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


def turn_mcp_interaction_text(turn) -> str:
    mcp_interaction = "Tools called by agent: \n"

    tools_called = turn.mcp_tools_called or turn.tools_called
    if tools_called:
        for tool in tools_called:
            if isinstance(tool, MCPToolCall):
                args = tool.args
                result = tool.result.structuredContent["result"]
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
