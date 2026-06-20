"""Shared string builders for MCPTaskCompletionMetric Jinja templates."""

from __future__ import annotations

from typing import List

from deepeval.metrics.mcp.schema import Task
from deepeval.test_case import MCPServer


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


def task_steps_taken_text(task: Task) -> str:
    return "\n\n".join(task.steps_taken)
