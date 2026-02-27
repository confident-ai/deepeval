from deepeval.tracing import observe
from tests.test_core.test_tracing.conftest import trace_test


@observe(type="tool", description="Search the web for information")
def web_search(query: str) -> str:
    return f"Search results for: {query}"


@observe(type="tool", description="Calculate mathematical expressions")
def calculator(expression: str) -> float:
    return 4.0


@observe(type="tool", name="custom_tool_name")
def tool_with_custom_name(data: str) -> str:
    return f"Processed: {data}"


@observe(type="tool")
def minimal_tool(input_data: str) -> str:
    return f"Tool output: {input_data}"


@observe(type="tool", description="Fetch data from API", name="api_fetcher")
def tool_with_description_and_name(url: str) -> dict:
    return {"url": url, "data": "fetched"}


@observe(
    type="tool",
    description="A very long description that explains what this tool does in great detail including all the parameters it accepts and the output format it returns",
)
def tool_with_long_description(data: str) -> str:
    return f"Processed: {data}"


class TestToolSpan:

    @trace_test("span_types/tool_span_schema.json")
    def test_tool_with_description(self):
        web_search("Python tutorials")

    @trace_test("span_types/tool_calculator_schema.json")
    def test_calculator_tool(self):
        calculator("2 + 2")

    @trace_test("span_types/tool_custom_name_schema.json")
    def test_tool_with_custom_name(self):
        tool_with_custom_name("test data")

    @trace_test("span_types/tool_minimal_schema.json")
    def test_minimal_tool(self):
        minimal_tool("input")

    @trace_test("span_types/tool_description_and_name_schema.json")
    def test_tool_with_description_and_name(self):
        tool_with_description_and_name("https://api.example.com")

    @trace_test("span_types/tool_long_description_schema.json")
    def test_tool_with_long_description(self):
        tool_with_long_description("data")
