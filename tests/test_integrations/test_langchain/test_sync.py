"""
Sync LangChain Tests
All synchronous tests using ChatOpenAI with deterministic settings.
"""

import os
from langchain_core.messages import HumanMessage
from deepeval.integrations.langchain import CallbackHandler
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
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
from tests.test_integrations.test_langchain.apps.langchain_metric_collection_app import (
    invoke_metric_collection_app,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")


def trace_test(schema_name: str):
    """
    Decorator that switches between generate and assert mode based on GENERATE_SCHEMAS env var.

    Args:
        schema_name: Name of the schema file (without path)
    """
    schema_path = os.path.join(_schemas_dir, schema_name)
    if is_generate_mode():
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
        """Test a simple greeting that returns a response."""
        callback = CallbackHandler(
            name="langchain-simple-test",
            tags=["langchain", "simple"],
            metadata={"test_type": "simple"},
            thread_id="simple-123",
            user_id="test-user",
        )

        result = invoke_simple_app(
            [HumanMessage(content="Say hello in one short sentence.")],
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


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
                    HumanMessage(
                        content="Use the get_weather tool to get weather for San Francisco. Do not ask clarifying questions."
                    )
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
        """Test query that uses one of the available city info tools."""
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
                        content="Use the get_weather tool to get the weather for Tokyo. Do not ask clarifying questions."
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_multiple_tools_mixed_schema.json")
    def test_mixed_query(self):
        """Test query that uses the weather tool."""
        callback = CallbackHandler(
            name="langchain-mixed-tools-test",
            tags=["langchain", "mixed-tools"],
            metadata={"test_type": "mixed_tools"},
        )

        result = invoke_mixed_tools(
            {
                "messages": [
                    HumanMessage(
                        content="Use the get_weather tool to get weather in Paris. Do not ask clarifying questions."
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
                    HumanMessage(
                        content="Use the get_stock_price tool to get the stock price for MSFT. Do not ask clarifying questions."
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_streaming_multi_schema.json")
    def test_sync_streaming_multiple_tools(self):
        """Test sync streaming with stock price tool."""
        callback = CallbackHandler(
            name="langchain-streaming-multi",
            tags=["langchain", "streaming", "multi-tool"],
        )

        result = invoke_streaming_multi(
            {
                "messages": [
                    HumanMessage(
                        content="Use the get_stock_price tool to get the stock price for TSLA. Do not ask clarifying questions."
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
            {
                "messages": [
                    HumanMessage(
                        content="Use the research_topic tool to research quantum computing. Do not ask clarifying questions."
                    )
                ]
            },
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
                        content="Use the summarize_text tool to summarize this: AI is transforming industries worldwide. Do not ask clarifying questions."
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
                    HumanMessage(
                        content="Use the fact_check tool to fact check this claim: The earth is round. Do not ask clarifying questions."
                    )
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
            {
                "messages": [
                    HumanMessage(content="Say hello in one short sentence.")
                ]
            },
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
                        content="Use the get_weather tool to get weather for Tokyo, New York, and London. Make all calls. Do not ask clarifying questions."
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    @trace_test("langchain_parallel_mixed_schema.json")
    def test_parallel_mixed_tools(self):
        """Test parallel execution with weather tool."""
        callback = CallbackHandler(
            name="langchain-parallel-mixed",
            tags=["langchain", "parallel", "mixed"],
        )

        result = invoke_parallel_mixed(
            {
                "messages": [
                    HumanMessage(
                        content="Use the get_weather tool to get weather in Paris. Do not ask clarifying questions."
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
                        content="Use the get_stock_price tool to get the price for AAPL. Do not ask clarifying questions."
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
                    HumanMessage(
                        content="Tell me about Python programming language."
                    )
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
            {
                "messages": [
                    HumanMessage(content="What is LangChain framework?")
                ]
            },
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
                    HumanMessage(
                        content="Use the search_web tool to search for 'weather san francisco'. Do not ask clarifying questions."
                    )
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
                        content="Use the search_web tool to search for 'stock price apple'. Do not ask clarifying questions."
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
                        content="Use the get_current_time tool to get the current time. Do not ask clarifying questions."
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# METRIC COLLECTION TESTS
# =============================================================================


class TestMetricCollectionApp:
    """Tests for metric_collection on LLM and tool spans."""

    @trace_test("langchain_metric_collection_schema.json")
    def test_metric_collection(self):
        """Test metric_collection on LLM and tool spans with prompt tracking."""
        callback = CallbackHandler(
            name="langchain-metric-collection",
            tags=["langchain", "metric-collection"],
            metadata={"test_type": "metric_collection"},
            metric_collection="trace_quality",
        )

        result = invoke_metric_collection_app(
            {
                "messages": [
                    HumanMessage(
                        content="Use the calculate tool to compute 15 * 3. Do not ask clarifying questions."
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0
