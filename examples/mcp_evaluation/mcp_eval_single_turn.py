import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from anthropic import Anthropic
from dotenv import load_dotenv

from deepeval.test_case import MCPServer, MCPToolCall, LLMTestCase

load_dotenv()

mcp_servers = []
tools_called = []


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(
        self, base_url: str, api_key: str, profile: str
    ):
        from urllib.parse import urlencode

        params = {"api_key": api_key, "profile": profile}
        url = f"{base_url}?{urlencode(params)}"

        transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(url)
        )
        read, write, _ = transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()

        tool_list = await self.session.list_tools()
        mcp_servers.append(
            MCPServer(
                server_name=base_url,
                available_tools=tool_list.tools,
            )
        )

    async def process_query(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]

        response_text = []

        tool_response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tool_response.tools
        ]

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )

        tool_uses = []

        for content in response.content:
            if content.type == "text":
                response_text.append(content.text)
            elif content.type == "tool_use":
                tool_uses.append(content)

        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_args = tool_use.input
            tool_id = tool_use.id

            result = await self.session.call_tool(tool_name, tool_args)
            tool_called = MCPToolCall(
                name=tool_name, args=tool_args, result=result
            )

            tools_called.append(tool_called)

        return "\n".join(response_text)

    async def chat_loop(self):

        query = input("Query: ")
        response = await self.process_query(query)

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            mcp_servers=mcp_servers,
            mcp_tools_called=tools_called,
        )

        print(test_case)

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 3:
        print("Usage: python client.py <api_key> <profile>")
        sys.exit(1)

    base_url = "https://your-server-url.mcp/github/mcp"
    api_key = "Your-api-key"
    profile = "Your-profile"

    client = MCPClient()
    try:
        await client.connect_to_server(base_url, api_key, profile)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
