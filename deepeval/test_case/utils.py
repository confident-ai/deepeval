from typing import Union, List, Optional, Set

from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    ToolCall,
    ToolCallType,
    MCPServer,
    get_available_mcp_tool_names,
    normalize_mcp_servers,
)


def check_valid_test_cases_type(
    test_cases: Union[List[LLMTestCase], List[ConversationalTestCase]],
):
    llm_test_case_count = 0
    conversational_test_case_count = 0
    for test_case in test_cases:
        if isinstance(test_case, LLMTestCase):
            llm_test_case_count += 1
        else:
            conversational_test_case_count += 1

    if llm_test_case_count > 0 and conversational_test_case_count > 0:
        raise ValueError(
            "You cannot supply a mixture of `LLMTestCase`(s) and `ConversationalTestCase`(s) as the list of test cases."
        )


def classify_tool_call_types(
    tools_called: Optional[List[ToolCall]],
    mcp_tool_names: Set[str],
):
    for tool_called in tools_called or []:
        if tool_called.name in mcp_tool_names:
            tool_called.type = ToolCallType.MCP


def process_mcp_servers(
    test_cases: Union[List[LLMTestCase], List[ConversationalTestCase]],
    mcp_servers: Optional[List[MCPServer]] = None,
):
    if mcp_servers is not None:
        mcp_servers = normalize_mcp_servers(mcp_servers)

    for test_case in test_cases:
        servers = test_case.mcp_servers or mcp_servers
        if not servers:
            continue

        test_case.mcp_servers = servers
        mcp_tool_names = get_available_mcp_tool_names(servers)
        if not mcp_tool_names:
            continue

        if isinstance(test_case, ConversationalTestCase):
            for turn in test_case.turns:
                classify_tool_call_types(turn.tools_called, mcp_tool_names)
        else:
            classify_tool_call_types(test_case.tools_called, mcp_tool_names)
            classify_tool_call_types(test_case.expected_tools, mcp_tool_names)
