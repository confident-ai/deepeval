"""Evaluate a public X search host through Xquik's remote MCP server.

Set XQUIK_API_KEY and ANTHROPIC_API_KEY before running this file. Set
ANTHROPIC_MODEL to override the default model.
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Optional
from urllib.parse import urlsplit

from anthropic import AsyncAnthropic

from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall


XQUIK_MCP_URL = "https://xquik.com/mcp"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOOL_ROUNDS = 4
READ_ONLY_SYSTEM_PROMPT = (
    "Use the available MCP tools only for public X search. "
    "Call GET operations only. Never call a write or billing operation."
)


def _bearer_headers(api_key: str) -> dict[str, str]:
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("XQUIK_API_KEY cannot be empty")
    return {"Authorization": f"Bearer {api_key}"}


def _validate_server_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("MCP server URL must use HTTPS")
    if parsed.query or parsed.fragment:
        raise ValueError("MCP server URL cannot contain a query or fragment")


def _serialize_tool_result(result: object) -> str:
    model_dump_json = getattr(result, "model_dump_json", None)
    if callable(model_dump_json):
        return model_dump_json(by_alias=True, exclude_none=True)
    return json.dumps(result, default=str)


def _required_environment_variable(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before running this example")
    return value


class MCPClient:
    def __init__(
        self,
        anthropic_client: Optional[Any] = None,
        model: str = DEFAULT_MODEL,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ):
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        self.session: Optional[Any] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = anthropic_client or AsyncAnthropic()
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.available_tools: list[Any] = []
        self.mcp_servers: list[MCPServer] = []

    async def connect_to_server(self, url: str, api_key: str) -> None:
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        _validate_server_url(url)
        http_client = await self.exit_stack.enter_async_context(
            httpx.AsyncClient(
                headers=_bearer_headers(api_key),
                timeout=httpx.Timeout(30, read=300),
            )
        )
        read, write, _ = await self.exit_stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()

        tool_list = await self.session.list_tools()
        self.available_tools = tool_list.tools
        self.mcp_servers = [
            MCPServer(
                server_name="Xquik",
                transport="streamable-http",
                available_tools=self.available_tools,
            )
        ]

    def _anthropic_tools(self) -> list[dict[str, object]]:
        tools = []
        for tool in self.available_tools:
            if isinstance(tool, dict):
                name = tool["name"]
                description = tool.get("description") or ""
                input_schema = tool["inputSchema"]
            else:
                name = tool.name
                description = tool.description or ""
                input_schema = tool.inputSchema
            tools.append(
                {
                    "name": name,
                    "description": description,
                    "input_schema": input_schema,
                }
            )
        return tools

    async def process_query(self, query: str) -> tuple[str, list[MCPToolCall]]:
        if self.session is None or not self.available_tools:
            raise RuntimeError("Connect to the MCP server first")

        messages: list[dict[str, object]] = [{"role": "user", "content": query}]
        tools_called: list[MCPToolCall] = []

        tool_rounds = 0
        while True:
            response = await self.anthropic.messages.create(
                model=self.model,
                max_tokens=1024,
                system=READ_ONLY_SYSTEM_PROMPT,
                messages=messages,
                tools=self._anthropic_tools(),
            )
            messages.append({"role": "assistant", "content": response.content})

            response_text = []
            tool_uses = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    response_text.append(block.text.strip())
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if not tool_uses:
                actual_output = "\n".join(response_text).strip()
                if not actual_output:
                    raise RuntimeError("The agent returned no visible output")
                return actual_output, tools_called

            if tool_rounds >= self.max_tool_rounds:
                raise RuntimeError(
                    "The agent exceeded the MCP tool round limit"
                )

            tool_results = []
            for tool_use in tool_uses:
                result = await self.session.call_tool(
                    tool_use.name, tool_use.input
                )
                tools_called.append(
                    MCPToolCall(
                        name=tool_use.name,
                        args=tool_use.input,
                        result=result,
                    )
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": _serialize_tool_result(result),
                        "is_error": bool(getattr(result, "isError", False)),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            tool_rounds += 1

    async def create_test_case(self, query: str) -> LLMTestCase:
        actual_output, tools_called = await self.process_query(query)
        return LLMTestCase(
            input=query,
            actual_output=actual_output,
            mcp_servers=self.mcp_servers,
            mcp_tools_called=tools_called,
        )

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()


async def main() -> None:
    client = MCPClient(model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))
    try:
        await client.connect_to_server(
            XQUIK_MCP_URL,
            _required_environment_variable("XQUIK_API_KEY"),
        )
        query = input("Public X search to evaluate: ").strip()
        if not query:
            raise RuntimeError("Provide a search task")
        print(await client.create_test_case(query))
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
