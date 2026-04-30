from deepeval.tracing import observe
from tests.test_core.test_tracing.conftest import trace_test


@observe(type="agent", available_tools=["search", "calculator"])
def simple_agent(query: str) -> str:
    return f"Agent processed: {query}"


@observe(
    type="agent",
    available_tools=["research", "summarize"],
    agent_handoffs=["writer_agent", "reviewer_agent"],
)
def agent_with_handoffs(query: str) -> str:
    return f"Agent with handoffs processed: {query}"


@observe(type="agent")
def minimal_agent(query: str) -> str:
    return f"Minimal agent: {query}"


@observe(type="agent", available_tools=["tool1"], name="custom_agent_name")
def agent_with_custom_name(query: str) -> str:
    return f"Named agent: {query}"


@observe(type="agent", agent_handoffs=["agent_a", "agent_b", "agent_c"])
def agent_multiple_handoffs(query: str) -> str:
    return f"Multi-handoff agent: {query}"


@observe(
    type="agent",
    available_tools=["search", "calculate", "fetch", "store"],
    agent_handoffs=["supervisor"],
)
def agent_full_attributes(query: str) -> str:
    return f"Full attributes agent: {query}"


class TestAgentSpan:

    @trace_test("span_types/agent_span_schema.json")
    def test_agent_with_tools(self):
        simple_agent("What is 2+2?")

    @trace_test("span_types/agent_with_handoffs_schema.json")
    def test_agent_with_handoffs(self):
        agent_with_handoffs("Research this topic")

    @trace_test("span_types/agent_minimal_schema.json")
    def test_minimal_agent(self):
        minimal_agent("Simple query")

    @trace_test("span_types/agent_custom_name_schema.json")
    def test_agent_with_custom_name(self):
        agent_with_custom_name("Test")

    @trace_test("span_types/agent_multiple_handoffs_schema.json")
    def test_agent_multiple_handoffs(self):
        agent_multiple_handoffs("Query")

    @trace_test("span_types/agent_full_attributes_schema.json")
    def test_agent_full_attributes(self):
        agent_full_attributes("Complex task")
