"""
Sync LangChain Tests
All synchronous tests using deterministic fake LLMs and tools.
"""

import os
from langchain_core.messages import HumanMessage
from deepeval.integrations.langchain import CallbackHandler
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
)

# App imports
from tests.test_integrations.test_langchain.apps.langchain_simple_app import (
    invoke_simple_app,
)
from tests.test_integrations.test_langchain.apps.langchain_single_tool_app import (
    invoke_single_tool_app,
)
from tests.test_integrations.test_langchain.apps.langchain_multiple_tools_app import (
    invoke_city_info,
    invoke_mixed_tools,
)
from tests.test_integrations.test_langchain.apps.langchain_streaming_app import (
    invoke_streaming_single,
    invoke_streaming_multi,
)
from tests.test_integrations.test_langchain.apps.langchain_conditional_app import (
    invoke_research,
    invoke_summarize,
    invoke_fact_check,
    invoke_general,
)
from tests.test_integrations.test_langchain.apps.langchain_parallel_tools_app import (
    invoke_parallel_weather,
    invoke_parallel_mixed,
    invoke_parallel_stocks,
)
from tests.test_integrations.test_langchain.apps.langchain_retriever_app import (
    invoke_rag_app,
)
from tests.test_integrations.test_langchain.apps.langchain_agent_app import (
    invoke_simple_agent,
    invoke_multi_step_agent,
    invoke_complex_agent,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set to True to generate schemas, False to assert against existing schemas
# Can be overridden via environment variable: GENERATE_SCHEMAS=true pytest ...
GENERATE_MODE = os.environ.get("GENERATE_SCHEMAS", "").lower() in (
    "true",
    "1",
    "yes",
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")


def trace_test(schema_name: str):
    """
    Decorator that switches between generate and assert mode based on GENERATE_MODE.

    Args:
        schema_name: Name of the schema file (without path)
    """
    schema_path = os.path.join(_schemas_dir, schema_name)
    if GENERATE_MODE:
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)


# =============================================================================
# SIMPLE APP TESTS (LLM only, no tools)
# =============================================================================


class TestSimpleApp:
    """Tests for simple LLM-only LangChain app."""

    @trace_test("langchain_simple_schema.json")
    def test_simple_greeting(self):
        """Test a simple greeting that returns a fixed response."""
        callback = CallbackHandler(
            name="langchain-simple-test",
            tags=["langchain", "simple"],
            metadata={"test_type": "simple"},
            thread_id="simple-123",
            user_id="test-user",
        )

        result = invoke_simple_app(
            [HumanMessage(content="Hello, how are you?")],
            config={"callbacks": [callback]},
        )

        assert result is not None
        assert hasattr(result, "content")
        assert "Hello" in result.content or "help" in result.content


# =============================================================================
# SINGLE TOOL TESTS
# =============================================================================


class TestSingleToolApp:
    """Tests for single-tool LangChain app."""

    @trace_test("langchain_single_tool_schema.json")
    def test_weather_query(self):
        """Test a simple weather query that triggers one tool call."""
        callback = CallbackHandler(
            name="langchain-single-tool-test",
            tags=["langchain", "single-tool"],
            metadata={"test_type": "single_tool"},
            thread_id="single-tool-123",
            user_id="test-user",
        )

        result = invoke_single_tool_app(
            {
                "messages": [
                    HumanMessage(content="What's the weather in San Francisco?")
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# MULTIPLE TOOLS TESTS
# =============================================================================


class TestMultipleToolsApp:
    """Tests for multi-tool LangChain app."""

    @trace_test("langchain_multiple_tools_schema.json")
    def test_city_info(self):
        """Test query that requires multiple tools about a city."""
        callback = CallbackHandler(
            name="langchain-multi-tool-test",
            tags=["langchain", "multiple-tools"],
            metadata={"test_type": "multiple_tools"},
            thread_id="multi-tool-123",
            user_id="test-user",
        )

        result = invoke_city_info(
            {
                "messages": [
                    HumanMessage(
                        content="Tell me about Tokyo - weather, population, timezone"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_multiple_tools_mixed_schema.json")
    def test_mixed_query(self):
        """Test query that requires mixed tool types (info + calculation)."""
        callback = CallbackHandler(
            name="langchain-mixed-tools-test",
            tags=["langchain", "mixed-tools"],
            metadata={"test_type": "mixed_tools"},
        )

        result = invoke_mixed_tools(
            {
                "messages": [
                    HumanMessage(
                        content="Weather in Paris and calculate 100 * 1.5 + 50"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# STREAMING TESTS
# =============================================================================


class TestStreamingApp:
    """Tests for streaming LangChain app."""

    @trace_test("langchain_streaming_schema.json")
    def test_sync_streaming(self):
        """Test sync streaming with tool calls."""
        callback = CallbackHandler(
            name="langchain-streaming-sync",
            tags=["langchain", "streaming", "sync"],
            metadata={"test_type": "streaming_sync"},
        )

        result = invoke_streaming_single(
            {
                "messages": [
                    HumanMessage(content="What's the stock price of MSFT?")
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_streaming_multi_schema.json")
    def test_sync_streaming_multiple_tools(self):
        """Test sync streaming with multiple tool calls."""
        callback = CallbackHandler(
            name="langchain-streaming-multi",
            tags=["langchain", "streaming", "multi-tool"],
        )

        result = invoke_streaming_multi(
            {
                "messages": [
                    HumanMessage(
                        content="Get stock price and company info for TSLA"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# CONDITIONAL ROUTING TESTS
# =============================================================================


class TestConditionalApp:
    """Tests for conditional routing LangChain app."""

    @trace_test("langchain_conditional_research_schema.json")
    def test_research_route(self):
        """Test routing to research tool."""
        callback = CallbackHandler(
            name="langchain-conditional-research",
            tags=["langchain", "conditional", "research"],
            metadata={"test_type": "conditional_research"},
        )

        result = invoke_research(
            {"messages": [HumanMessage(content="Research quantum computing")]},
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_conditional_summarize_schema.json")
    def test_summarize_route(self):
        """Test routing to summarize tool."""
        callback = CallbackHandler(
            name="langchain-conditional-summarize",
            tags=["langchain", "conditional", "summarize"],
        )

        result = invoke_summarize(
            {
                "messages": [
                    HumanMessage(
                        content="Summarize this: AI is transforming industries."
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_conditional_fact_check_schema.json")
    def test_fact_check_route(self):
        """Test routing to fact check tool."""
        callback = CallbackHandler(
            name="langchain-conditional-factcheck",
            tags=["langchain", "conditional", "fact-check"],
        )

        result = invoke_fact_check(
            {
                "messages": [
                    HumanMessage(content="Fact check: The earth is round")
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_conditional_general_schema.json")
    def test_general_route(self):
        """Test routing to general response (no tools)."""
        callback = CallbackHandler(
            name="langchain-conditional-general",
            tags=["langchain", "conditional", "general"],
        )

        result = invoke_general(
            {"messages": [HumanMessage(content="Hello, how are you today?")]},
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# PARALLEL TOOLS TESTS
# =============================================================================


class TestParallelToolsApp:
    """Tests for parallel tool execution LangChain app."""

    @trace_test("langchain_parallel_weather_schema.json")
    def test_parallel_weather_queries(self):
        """Test parallel weather queries for multiple cities."""
        callback = CallbackHandler(
            name="langchain-parallel-weather",
            tags=["langchain", "parallel", "weather"],
            metadata={"test_type": "parallel_weather"},
        )

        result = invoke_parallel_weather(
            {
                "messages": [
                    HumanMessage(
                        content="Weather in Tokyo, New York, and London?"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_parallel_mixed_schema.json")
    def test_parallel_mixed_tools(self):
        """Test parallel execution of different tool types."""
        callback = CallbackHandler(
            name="langchain-parallel-mixed",
            tags=["langchain", "parallel", "mixed"],
        )

        result = invoke_parallel_mixed(
            {
                "messages": [
                    HumanMessage(
                        content="Weather Paris, TSLA stock, USD to EUR, calc 100*1.5"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_parallel_stocks_schema.json")
    def test_parallel_stock_queries(self):
        """Test parallel stock price queries."""
        callback = CallbackHandler(
            name="langchain-parallel-stocks",
            tags=["langchain", "parallel", "stocks"],
        )

        result = invoke_parallel_stocks(
            {
                "messages": [
                    HumanMessage(
                        content="Stock prices for AAPL, GOOGL, MSFT, TSLA, AMZN"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# RETRIEVER (RAG) TESTS
# =============================================================================


class TestRetrieverApp:
    """Tests for RAG LangChain app with retriever."""

    @trace_test("langchain_retriever_python_schema.json")
    def test_retrieve_python_docs(self):
        """Test retrieval of Python-related documents."""
        callback = CallbackHandler(
            name="langchain-retriever-python",
            tags=["langchain", "retriever", "python"],
            metadata={"test_type": "retriever"},
        )

        result = invoke_rag_app(
            {
                "messages": [
                    HumanMessage(content="Tell me about Python programming")
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_retriever_langchain_schema.json")
    def test_retrieve_langchain_docs(self):
        """Test retrieval of LangChain-related documents."""
        callback = CallbackHandler(
            name="langchain-retriever-langchain",
            tags=["langchain", "retriever", "langchain-docs"],
        )

        result = invoke_rag_app(
            {"messages": [HumanMessage(content="What is LangChain?")]},
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# AGENT TESTS
# =============================================================================


class TestAgentApp:
    """Tests for agent-style LangChain app."""

    @trace_test("langchain_agent_simple_schema.json")
    def test_simple_agent(self):
        """Test simple agent with one tool call."""
        callback = CallbackHandler(
            name="langchain-agent-simple",
            tags=["langchain", "agent", "simple"],
            metadata={"test_type": "agent"},
        )

        result = invoke_simple_agent(
            {
                "messages": [
                    HumanMessage(content="What's the weather in San Francisco?")
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_agent_multi_step_schema.json")
    def test_multi_step_agent(self):
        """Test agent that makes multiple sequential tool calls."""
        callback = CallbackHandler(
            name="langchain-agent-multi-step",
            tags=["langchain", "agent", "multi-step"],
        )

        result = invoke_multi_step_agent(
            {
                "messages": [
                    HumanMessage(
                        content="What's Apple stock price and value of 100 shares?"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_agent_complex_schema.json")
    def test_complex_agent(self):
        """Test agent with complex multi-tool workflow."""
        callback = CallbackHandler(
            name="langchain-agent-complex",
            tags=["langchain", "agent", "complex"],
        )

        result = invoke_complex_agent(
            {
                "messages": [
                    HumanMessage(
                        content="What time is it, USD to EUR rate, and convert 1000 USD"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0
