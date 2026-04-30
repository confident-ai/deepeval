from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase, ToolCall
from tests.test_core.test_tracing.conftest import trace_test


@observe()
def span_update_input_output(data: str) -> str:
    update_current_span(
        input="Custom input override",
        output="Custom output override",
    )
    return f"Result: {data}"


@observe()
def span_update_context(query: str) -> str:
    update_current_span(
        retrieval_context=["Document 1 content", "Document 2 content"],
        context=["Additional context 1", "Additional context 2"],
    )
    return f"Contextualized: {query}"


@observe()
def span_update_expected_output(query: str) -> str:
    update_current_span(expected_output="Expected response format")
    return f"Response: {query}"


@observe()
def span_update_tools(query: str) -> str:
    update_current_span(
        tools_called=[
            ToolCall(name="search", input_parameters={"query": query}),
            ToolCall(name="calculate", input_parameters={"expr": "2+2"}),
        ],
        expected_tools=[ToolCall(name="search")],
    )
    return f"Tools used for: {query}"


@observe()
def span_update_name(data: str) -> str:
    update_current_span(name="custom_span_name")
    return data


@observe()
def span_from_test_case(data: str) -> str:
    test_case = LLMTestCase(
        input="Test case input",
        actual_output="Test case output",
        expected_output="Expected output",
        retrieval_context=["Context from test case"],
    )
    update_current_span(test_case=test_case)
    return data


@observe()
def span_override_test_case(data: str) -> str:
    test_case = LLMTestCase(
        input="Original input",
        actual_output="Original output",
        expected_output="Original expected",
    )
    update_current_span(test_case=test_case)
    update_current_span(expected_output="Overridden expected output")
    return data


class TestUpdateCurrentSpan:

    @trace_test("update_functions/span_input_output_schema.json")
    def test_update_input_output(self):
        span_update_input_output("test")

    @trace_test("update_functions/span_context_schema.json")
    def test_update_context(self):
        span_update_context("query")

    @trace_test("update_functions/span_expected_output_schema.json")
    def test_update_expected_output(self):
        span_update_expected_output("test query")

    @trace_test("update_functions/span_tools_schema.json")
    def test_update_tools(self):
        span_update_tools("search query")

    @trace_test("update_functions/span_name_schema.json")
    def test_update_name(self):
        span_update_name("data")

    @trace_test("update_functions/span_from_test_case_schema.json")
    def test_from_test_case(self):
        span_from_test_case("data")

    @trace_test("update_functions/span_override_test_case_schema.json")
    def test_override_test_case(self):
        span_override_test_case("data")
