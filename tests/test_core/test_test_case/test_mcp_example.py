from copy import deepcopy
from types import SimpleNamespace

import pytest

from deepeval.test_case import MCPServer
from examples.mcp_evaluation.mcp_eval_single_turn import (
    MCPClient,
    READ_ONLY_SYSTEM_PROMPT,
    _bearer_headers,
    _validate_server_url,
)


class FakeToolResult:
    def __init__(self, value: str):
        self.value = value
        self.isError = False

    def model_dump_json(self, **_: object) -> str:
        return f'{{"value":"{self.value}"}}'


class FakeSession:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict) -> FakeToolResult:
        self.calls.append((name, args))
        return FakeToolResult(name)


class FakeMessages:
    def __init__(self, responses: list[SimpleNamespace]):
        self.responses = responses
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(deepcopy(kwargs))
        return self.responses.pop(0)


class FakeAnthropic:
    def __init__(self, responses: list[SimpleNamespace]):
        self.messages = FakeMessages(responses)


def response(*blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks))


def tool_use(name: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use",
        name=name,
        id=call_id,
        input={"code": f"run {name}"},
    )


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def connected_client(
    responses: list[SimpleNamespace], max_tool_rounds: int = 4
) -> tuple[MCPClient, FakeSession]:
    client = MCPClient(
        anthropic_client=FakeAnthropic(responses),
        max_tool_rounds=max_tool_rounds,
    )
    session = FakeSession()
    tools = [
        {
            "name": "explore",
            "description": "Inspect the API catalog",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "xquik",
            "description": "Run API requests",
            "inputSchema": {"type": "object"},
        },
    ]
    client.session = session
    client.available_tools = tools
    client.mcp_servers = [
        MCPServer(
            server_name="Xquik",
            transport="streamable-http",
            available_tools=tools,
        )
    ]
    return client, session


def test_bearer_credentials_stay_out_of_the_url():
    assert _bearer_headers("  xq_example  ") == {
        "Authorization": "Bearer xq_example"
    }
    _validate_server_url("https://xquik.com/mcp")

    with pytest.raises(ValueError, match="query or fragment"):
        _validate_server_url("https://xquik.com/mcp?api_key=xq_example")


@pytest.mark.asyncio
async def test_returns_the_final_answer_after_all_tool_results():
    client, session = connected_client(
        [
            response(text("Searching..."), tool_use("explore", "call-1")),
            response(tool_use("xquik", "call-2")),
            response(text("Found five relevant public posts.")),
        ]
    )

    actual_output, tools_called = await client.process_query(
        "Find five posts about evals"
    )

    assert actual_output == "Found five relevant public posts."
    assert [call.name for call in tools_called] == [
        "explore",
        "xquik",
    ]
    assert session.calls == [
        ("explore", {"code": "run explore"}),
        ("xquik", {"code": "run xquik"}),
    ]
    model_calls = client.anthropic.messages.calls
    assert all(
        call["system"] == READ_ONLY_SYSTEM_PROMPT for call in model_calls
    )
    assert model_calls[1]["messages"][-1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call-1",
        "content": '{"value":"explore"}',
        "is_error": False,
    }


@pytest.mark.asyncio
async def test_creates_a_test_case_with_official_mcp_results():
    mcp_types = pytest.importorskip("mcp.types")
    client, session = connected_client(
        [
            response(tool_use("explore", "call-1")),
            response(text("The search completed.")),
        ]
    )

    async def call_tool(name: str, args: dict):
        session.calls.append((name, args))
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=name)],
            structuredContent={"result": name},
            isError=False,
        )

    session.call_tool = call_tool

    test_case = await client.create_test_case("Search public posts")

    assert test_case.actual_output == "The search completed."
    assert test_case.mcp_tools_called[0].name == "explore"


@pytest.mark.asyncio
async def test_rejects_an_incomplete_tool_loop():
    client, _ = connected_client(
        [
            response(tool_use("explore", "call-1")),
            response(tool_use("xquik", "call-2")),
        ],
        max_tool_rounds=1,
    )

    with pytest.raises(RuntimeError, match="tool round limit"):
        await client.create_test_case("Search public posts")
