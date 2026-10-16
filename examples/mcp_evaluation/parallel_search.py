"""Capture anonymous Parallel search/fetch calls as a DeepEval test case."""

import argparse
import asyncio
from datetime import timedelta
from uuid import uuid4

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = (
    "deepeval-parallel-search-example/1.0 "
    "(https://github.com/confident-ai/deepeval)"
)


async def capture_search(
    objective: str, search_queries: list[str], urls: list[str]
) -> LLMTestCase:
    """Run tools directly, keeping discovery and calls in one MCP session."""
    session_id = str(uuid4())
    calls = []
    output = []
    async with streamablehttp_client(
        MCP_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
        sse_read_timeout=120,
    ) as (read, write, _):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=120)
        ) as session:
            await session.initialize()
            tools = await session.list_tools()
            server = MCPServer(
                server_name="Parallel Search",
                transport="streamable-http",
                available_tools=tools.tools,
            )
            requests = [
                (
                    "web_search",
                    {
                        "objective": objective,
                        "search_queries": search_queries,
                        "session_id": session_id,
                    },
                )
            ]
            if urls:
                requests.append(
                    (
                        "web_fetch",
                        {"urls": urls, "session_id": session_id},
                    )
                )
            available = {tool.name for tool in tools.tools}
            for name, args in requests:
                if name not in available:
                    raise RuntimeError(f"MCP server does not offer {name}")
                result = await session.call_tool(name, args)
                if result.isError:
                    raise RuntimeError(f"{name} failed: {result.content}")
                text = "\n".join(
                    block.text
                    for block in result.content
                    if isinstance(block, TextContent)
                )
                if not text.strip():
                    raise RuntimeError(f"{name} returned no text")
                calls.append(MCPToolCall(name=name, args=args, result=result))
                output.append(f"{name}:\n{text}")

    return LLMTestCase(
        input=objective,
        actual_output="\n\n".join(output),
        mcp_servers=[server],
        mcp_tools_called=calls,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objective", help="What to find on the web")
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Search query; repeatable",
    )
    parser.add_argument(
        "--url", action="append", default=[], help="Page to fetch; repeatable"
    )
    args = parser.parse_args()
    if len(args.url) > 20:
        parser.error("--url accepts at most 20 pages per fetch")
    test_case = asyncio.run(
        capture_search(args.objective, args.query, args.url)
    )
    print(test_case.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
