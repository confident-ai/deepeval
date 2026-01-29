from deepeval.tracing import observe, update_current_trace
from deepeval.test_case import LLMTestCase, ToolCall
from tests.test_core.test_tracing.conftest import trace_test


@observe()
def trace_update_name(data: str) -> str:
    update_current_trace(name="custom_trace_name")
    return f"Named trace: {data}"


@observe()
def trace_update_identifiers(data: str) -> str:
    update_current_trace(
        user_id="user_123",
        thread_id="thread_456",
    )
    return f"Identified: {data}"


@observe()
def trace_update_all_context(query: str) -> str:
    update_current_trace(
        name="full_context_trace",
        tags=["test", "full"],
        metadata={"version": "1.0", "env": "test"},
        user_id="user_001",
        thread_id="thread_001",
        input="Custom trace input",
        output="Custom trace output",
    )
    return f"Full context: {query}"


@observe()
def trace_update_context_info(query: str) -> str:
    update_current_trace(
        retrieval_context=["Trace-level doc 1", "Trace-level doc 2"],
        context=["Additional trace context"],
        expected_output="Expected trace output",
    )
    return f"Context set: {query}"


@observe()
def trace_update_tools(query: str) -> str:
    update_current_trace(
        tools_called=[ToolCall(name="search", output="Search results")],
        expected_tools=[ToolCall(name="search"), ToolCall(name="summarize")],
    )
    return f"Trace tools: {query}"


@observe()
def trace_from_test_case(data: str) -> str:
    test_case = LLMTestCase(
        input="Trace test input",
        actual_output="Trace test output",
        expected_output="Trace expected output",
        context=["Test context"],
    )
    update_current_trace(test_case=test_case)
    return data


@observe()
def outer_sets_trace_context(data: str) -> str:
    update_current_trace(name="outer_set_name", user_id="outer_user")
    return inner_reads_context(data)


@observe()
def inner_reads_context(data: str) -> str:
    update_current_trace(tags=["inner_added"])
    return f"Inner: {data}"


class TestUpdateCurrentTrace:

    @trace_test("update_functions/trace_name_schema.json")
    def test_update_name(self):
        trace_update_name("test")

    @trace_test("update_functions/trace_identifiers_schema.json")
    def test_update_identifiers(self):
        trace_update_identifiers("test")

    @trace_test("update_functions/trace_full_context_schema.json")
    def test_update_all_context(self):
        trace_update_all_context("query")

    @trace_test("update_functions/trace_context_info_schema.json")
    def test_update_context_info(self):
        trace_update_context_info("query")

    @trace_test("update_functions/trace_tools_schema.json")
    def test_update_tools(self):
        trace_update_tools("query")

    @trace_test("update_functions/trace_from_test_case_schema.json")
    def test_from_test_case(self):
        trace_from_test_case("data")

    @trace_test("update_functions/trace_nested_updates_schema.json")
    def test_nested_trace_updates(self):
        outer_sets_trace_context("test")
