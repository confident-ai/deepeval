from types import SimpleNamespace

from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    Turn,
    ToolCall,
    ToolCallType,
    MCPServer,
    get_available_mcp_tool_names,
)
from deepeval.evaluate.api import APIEvaluate
from deepeval.test_case.api import create_api_test_case
from deepeval.test_case.utils import process_mcp_servers


class TestGetAvailableMCPToolNames:

    def test_returns_names_from_dicts(self):
        mcp_servers = [
            MCPServer(
                server_name="GitHub",
                available_tools=[{"name": "search"}, {"name": "create_issue"}],
            )
        ]

        assert get_available_mcp_tool_names(mcp_servers) == {
            "search",
            "create_issue",
        }

    def test_returns_names_across_servers(self):
        mcp_servers = [
            MCPServer(
                server_name="GitHub", available_tools=[{"name": "search"}]
            ),
            MCPServer(
                server_name="Slack", available_tools=[{"name": "post_message"}]
            ),
        ]

        assert get_available_mcp_tool_names(mcp_servers) == {
            "search",
            "post_message",
        }

    def test_returns_names_from_objects(self):
        mcp_servers = [
            MCPServer(
                server_name="GitHub",
                available_tools=[SimpleNamespace(name="search")],
            )
        ]

        assert get_available_mcp_tool_names(mcp_servers) == {"search"}

    def test_returns_empty_set_when_no_available_tools(self):
        mcp_servers = [MCPServer(server_name="GitHub")]

        assert get_available_mcp_tool_names(mcp_servers) == set()

    def test_skips_tools_without_a_name(self):
        mcp_servers = [
            MCPServer(
                server_name="GitHub",
                available_tools=[
                    {"description": "no name"},
                    {"name": "search"},
                ],
            )
        ]

        assert get_available_mcp_tool_names(mcp_servers) == {"search"}


class TestProcessMCPServers:

    def test_tags_matching_tool_calls_as_mcp(self):
        test_case = LLMTestCase(
            input="Find the issue",
            actual_output="Found it",
            tools_called=[ToolCall(name="search"), ToolCall(name="local_fn")],
        )

        process_mcp_servers(
            [test_case],
            [
                MCPServer(
                    server_name="GitHub", available_tools=[{"name": "search"}]
                )
            ],
        )

        assert test_case.tools_called[0].type == ToolCallType.MCP
        assert test_case.tools_called[1].type == ToolCallType.FUNCTION

    def test_tags_expected_tools_as_mcp(self):
        test_case = LLMTestCase(
            input="Find the issue",
            actual_output="Found it",
            expected_tools=[ToolCall(name="search")],
        )

        process_mcp_servers(
            [test_case],
            [
                MCPServer(
                    server_name="GitHub", available_tools=[{"name": "search"}]
                )
            ],
        )

        assert test_case.expected_tools[0].type == ToolCallType.MCP

    def test_tags_conversational_turn_tool_calls_as_mcp(self):
        test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content="Find the issue"),
                Turn(
                    role="assistant",
                    content="Found it",
                    tools_called=[
                        ToolCall(name="search"),
                        ToolCall(name="local_fn"),
                    ],
                ),
            ]
        )

        process_mcp_servers(
            [test_case],
            [
                MCPServer(
                    server_name="GitHub", available_tools=[{"name": "search"}]
                )
            ],
        )

        assert test_case.turns[1].tools_called[0].type == ToolCallType.MCP
        assert test_case.turns[1].tools_called[1].type == ToolCallType.FUNCTION

    def test_assigns_mcp_servers_when_test_case_has_none(self):
        test_case = LLMTestCase(
            input="Find the issue", actual_output="Found it"
        )
        mcp_servers = [
            MCPServer(
                server_name="GitHub", available_tools=[{"name": "search"}]
            )
        ]

        process_mcp_servers([test_case], mcp_servers)

        assert test_case.mcp_servers == mcp_servers

    def test_test_case_mcp_servers_take_precedence(self):
        test_case = LLMTestCase(
            input="Find the issue",
            actual_output="Found it",
            mcp_servers=[
                MCPServer(
                    server_name="Local", available_tools=[{"name": "local_fn"}]
                )
            ],
            tools_called=[ToolCall(name="search"), ToolCall(name="local_fn")],
        )

        process_mcp_servers(
            [test_case],
            [
                MCPServer(
                    server_name="GitHub", available_tools=[{"name": "search"}]
                )
            ],
        )

        assert test_case.mcp_servers[0].server_name == "Local"
        assert test_case.tools_called[0].type == ToolCallType.FUNCTION
        assert test_case.tools_called[1].type == ToolCallType.MCP

    def test_no_op_when_no_mcp_servers(self):
        test_case = LLMTestCase(
            input="Find the issue",
            actual_output="Found it",
            tools_called=[ToolCall(name="search")],
        )

        process_mcp_servers([test_case], None)

        assert test_case.mcp_servers is None
        assert test_case.tools_called[0].type == ToolCallType.FUNCTION


class TestMCPToolCallTypeSerialization:

    def test_tool_call_type_is_sent_for_llm_test_case(self):
        test_case = LLMTestCase(
            input="Find the issue",
            actual_output="Found it",
            tools_called=[ToolCall(name="search"), ToolCall(name="local_fn")],
        )

        process_mcp_servers(
            [test_case],
            [
                MCPServer(
                    server_name="GitHub", available_tools=[{"name": "search"}]
                )
            ],
        )
        body = create_api_test_case(test_case).model_dump(
            by_alias=True, exclude_none=True
        )

        assert body["toolsCalled"] == [
            {"name": "search", "type": "MCP"},
            {"name": "local_fn", "type": "FUNCTION"},
        ]

    def test_tool_call_type_is_sent_for_conversational_test_case(self):
        test_case = ConversationalTestCase(
            turns=[
                Turn(
                    role="assistant",
                    content="Found it",
                    tools_called=[ToolCall(name="search")],
                )
            ]
        )

        process_mcp_servers(
            [test_case],
            [
                MCPServer(
                    server_name="GitHub", available_tools=[{"name": "search"}]
                )
            ],
        )
        body = create_api_test_case(test_case).model_dump(
            by_alias=True, exclude_none=True
        )

        assert body["turns"][0]["toolsCalled"] == [
            {"name": "search", "type": "MCP"}
        ]

    def test_tool_call_type_is_sent_for_metric_collection(self):
        test_case = LLMTestCase(
            input="Find the issue",
            actual_output="Found it",
            tools_called=[ToolCall(name="search"), ToolCall(name="local_fn")],
        )

        process_mcp_servers(
            [test_case],
            [
                MCPServer(
                    server_name="GitHub", available_tools=[{"name": "search"}]
                )
            ],
        )
        body = APIEvaluate(
            metricCollection="My Collection",
            llmTestCases=[test_case],
            conversationalTestCases=None,
        ).model_dump(by_alias=True, exclude_none=True)

        assert body["llmTestCases"][0]["toolsCalled"] == [
            {"name": "search", "type": "MCP"},
            {"name": "local_fn", "type": "FUNCTION"},
        ]
