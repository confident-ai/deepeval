"""
Example: Evaluate an AI agent using the TWZRD Agent Intel MCP server with DeepEval.

The TWZRD Agent Intel server (https://intel.twzrd.xyz) exposes trust-scoring tools
for AI agent wallets (score_agent, preflight_check) via MCP over streamable-HTTP.

This example:
1. Connects to the TWZRD Agent Intel MCP server
2. Asks Claude to run a preflight check on a test agent wallet
3. Records the MCP tool calls
4. Evaluates correctness with deepeval's MCPToolCorrectnessMetric

Run:
    pip install deepeval anthropic mcp python-dotenv
    ANTHROPIC_API_KEY=sk-... python mcp_eval_twzrd_agent_trust.py
"""
import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from anthropic import Anthropic

from deepeval.test_case import MCPServer, MCPToolCall, LLMTestCase
from deepeval.metrics import MCPToolCorrectnessMetric
from deepeval import evaluate

# ---------------------------------------------------------------------------
# MCP client (matches the pattern in mcp_eval_single_turn.py)
# ---------------------------------------------------------------------------

mcp_servers = []
tools_called = []

TWZRD_MCP_URL = "https://intel.twzrd.xyz/mcp"
TEST_WALLET = "D1QkbFJKiPsymJ65RKHhF6DFB8sPMfpBaFBzuHKfJGWi"


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect(self, url: str):
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
                server_name=url,
                available_tools=tool_list.tools,
            )
        )

    async def run_query(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        tool_response = await self.session.list_tools()
        available_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in tool_response.tools
        ]

        response = self.anthropic.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=messages,
            tools=available_tools,
        )

        result_parts = []
        for block in response.content:
            if block.type == "tool_use":
                tools_called.append(
                    MCPToolCall(
                        tool_name=block.name,
                        input=block.input,
                    )
                )
                tool_result = await self.session.call_tool(block.name, block.input)
                result_parts.append(str(tool_result.content))
            elif block.type == "text":
                result_parts.append(block.text)

        return " ".join(result_parts)

    async def close(self):
        await self.exit_stack.aclose()


async def run():
    client = MCPClient()
    await client.connect(TWZRD_MCP_URL)

    output = await client.run_query(
        f"Run a preflight_check for agent wallet {TEST_WALLET}. "
        "Report whether the agent passes or fails the trust check."
    )

    await client.close()
    return output


# ---------------------------------------------------------------------------
# DeepEval test
# ---------------------------------------------------------------------------

def main():
    actual_output = asyncio.run(run())
    print("Agent output:", actual_output)

    test_case = LLMTestCase(
        input=f"Run preflight_check for wallet {TEST_WALLET}",
        actual_output=actual_output,
        mcp_tool_calls=tools_called,
    )

    metric = MCPToolCorrectnessMetric(
        mcp_servers=mcp_servers,
        expected_tool_calls=[
            MCPToolCall(
                tool_name="preflight_check",
                input={"wallet": TEST_WALLET},
            )
        ],
    )

    evaluate(test_cases=[test_case], metrics=[metric])


if __name__ == "__main__":
    main()
