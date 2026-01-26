"""
Async LangChain Tests
All asynchronous tests using ainvoke() methods.
"""

import os
import pytest
from langchain_core.messages import HumanMessage
from deepeval.integrations.langchain import CallbackHandler
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

# App imports
from tests.test_integrations.test_langchain.apps.langchain_simple_app import (
    ainvoke_simple_app,
)
from tests.test_integrations.test_langchain.apps.langchain_single_tool_app import (
    ainvoke_single_tool_app,
)
from tests.test_integrations.test_langchain.apps.langchain_multiple_tools_app import (
    ainvoke_city_info,
    ainvoke_mixed_tools,
)
from tests.test_integrations.test_langchain.apps.langchain_streaming_app import (
    ainvoke_streaming_single,
    ainvoke_streaming_multi,
)
from tests.test_integrations.test_langchain.apps.langchain_conditional_app import (
    ainvoke_research,
    ainvoke_summarize,
    ainvoke_fact_check,
    ainvoke_general,
)
from tests.test_integrations.test_langchain.apps.langchain_parallel_tools_app import (
    ainvoke_parallel_weather,
    ainvoke_parallel_mixed,
    ainvoke_parallel_stocks,
)
from tests.test_integrations.test_langchain.apps.langchain_retriever_app import (
    ainvoke_rag_app,
)
from tests.test_integrations.test_langchain.apps.langchain_agent_app import (
    ainvoke_simple_agent,
    ainvoke_multi_step_agent,
    ainvoke_complex_agent,
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
# ASYNC SIMPLE APP TESTS
# =============================================================================


class TestAsyncSimpleApp:
    """Async tests for simple LLM-only LangChain app."""

    @pytest.mark.asyncio
    @trace_test("langchain_async_simple_schema.json")
    async def test_async_simple_greeting(self):
        """Test async simple greeting."""
        callback = CallbackHandler(
            name="langchain-async-simple",
            tags=["langchain", "async", "simple"],
            metadata={"test_type": "async_simple"},
            thread_id="async-simple-123",
            user_id="async-user",
        )

        result = await ainvoke_simple_app(
            [HumanMessage(content="Say hello in one short sentence.")],
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# ASYNC SINGLE TOOL TESTS
# =============================================================================


class TestAsyncSingleToolApp:
    """Async tests for single-tool LangChain app."""

    @pytest.mark.asyncio
    @trace_test("langchain_async_single_tool_schema.json")
    async def test_async_weather_query(self):
        """Test async weather query with single tool."""
        callback = CallbackHandler(
            name="langchain-async-single-tool",
            tags=["langchain", "async", "single-tool"],
            metadata={"test_type": "async_single_tool"},
            thread_id="async-single-tool-123",
            user_id="async-user",
        )

        result = await ainvoke_single_tool_app(
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
# ASYNC MULTIPLE TOOLS TESTS
# =============================================================================


class TestAsyncMultipleToolsApp:
    """Async tests for multi-tool LangChain app."""

    @pytest.mark.asyncio
    @trace_test("langchain_async_multiple_tools_schema.json")
    async def test_async_city_info(self):
        """Test async query with multiple tools about a city."""
        callback = CallbackHandler(
            name="langchain-async-multi-tool",
            tags=["langchain", "async", "multi-tool"],
            metadata={"test_type": "async_multi_tool"},
            thread_id="async-multi-tool-123",
            user_id="async-user",
        )

        result = await ainvoke_city_info(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_mixed_tools_schema.json")
    async def test_async_mixed_query(self):
        """Test async query with mixed tool types."""
        callback = CallbackHandler(
            name="langchain-async-mixed-tools",
            tags=["langchain", "async", "mixed-tools"],
        )

        result = await ainvoke_mixed_tools(
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
# ASYNC STREAMING TESTS
# =============================================================================


class TestAsyncStreamingApp:
    """Async tests for streaming LangChain app."""

    @pytest.mark.asyncio
    @trace_test("langchain_async_streaming_schema.json")
    async def test_async_streaming_single(self):
        """Test async streaming with single tool."""
        callback = CallbackHandler(
            name="langchain-async-streaming-single",
            tags=["langchain", "async", "streaming", "single"],
            metadata={"test_type": "async_streaming_single"},
        )

        result = await ainvoke_streaming_single(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_streaming_multi_schema.json")
    async def test_async_streaming_multi(self):
        """Test async streaming with multiple tools."""
        callback = CallbackHandler(
            name="langchain-async-streaming-multi",
            tags=["langchain", "async", "streaming", "multi"],
        )

        result = await ainvoke_streaming_multi(
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
# ASYNC CONDITIONAL ROUTING TESTS
# =============================================================================


class TestAsyncConditionalApp:
    """Async tests for conditional routing LangChain app."""

    @pytest.mark.asyncio
    @trace_test("langchain_async_conditional_research_schema.json")
    async def test_async_research_route(self):
        """Test async routing to research tool."""
        callback = CallbackHandler(
            name="langchain-async-conditional-research",
            tags=["langchain", "async", "conditional", "research"],
            metadata={"test_type": "async_conditional_research"},
        )

        result = await ainvoke_research(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_conditional_summarize_schema.json")
    async def test_async_summarize_route(self):
        """Test async routing to summarize tool."""
        callback = CallbackHandler(
            name="langchain-async-conditional-summarize",
            tags=["langchain", "async", "conditional", "summarize"],
        )

        result = await ainvoke_summarize(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_conditional_fact_check_schema.json")
    async def test_async_fact_check_route(self):
        """Test async routing to fact check tool."""
        callback = CallbackHandler(
            name="langchain-async-conditional-factcheck",
            tags=["langchain", "async", "conditional", "fact-check"],
        )

        result = await ainvoke_fact_check(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_conditional_general_schema.json")
    async def test_async_general_route(self):
        """Test async routing to general response (no tools)."""
        callback = CallbackHandler(
            name="langchain-async-conditional-general",
            tags=["langchain", "async", "conditional", "general"],
        )

        result = await ainvoke_general(
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
# ASYNC PARALLEL TOOLS TESTS
# =============================================================================


class TestAsyncParallelToolsApp:
    """Async tests for parallel tool execution LangChain app."""

    @pytest.mark.asyncio
    @trace_test("langchain_async_parallel_weather_schema.json")
    async def test_async_parallel_weather(self):
        """Test async parallel weather queries."""
        callback = CallbackHandler(
            name="langchain-async-parallel-weather",
            tags=["langchain", "async", "parallel", "weather"],
            metadata={"test_type": "async_parallel_weather"},
        )

        result = await ainvoke_parallel_weather(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_parallel_mixed_schema.json")
    async def test_async_parallel_mixed(self):
        """Test async parallel execution of different tool types."""
        callback = CallbackHandler(
            name="langchain-async-parallel-mixed",
            tags=["langchain", "async", "parallel", "mixed"],
        )

        result = await ainvoke_parallel_mixed(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_parallel_stocks_schema.json")
    async def test_async_parallel_stocks(self):
        """Test async parallel stock price queries."""
        callback = CallbackHandler(
            name="langchain-async-parallel-stocks",
            tags=["langchain", "async", "parallel", "stocks"],
        )

        result = await ainvoke_parallel_stocks(
            {
                "messages": [
                    HumanMessage(
                        content="Use the get_stock_price tool to get prices for AAPL. Do not ask clarifying questions."
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# ASYNC RETRIEVER (RAG) TESTS
# =============================================================================


class TestAsyncRetrieverApp:
    """Async tests for RAG LangChain app with retriever."""

    @pytest.mark.asyncio
    @trace_test("langchain_async_retriever_python_schema.json")
    async def test_async_retrieve_python_docs(self):
        """Test async retrieval of Python-related documents."""
        callback = CallbackHandler(
            name="langchain-async-retriever-python",
            tags=["langchain", "async", "retriever", "python"],
            metadata={"test_type": "async_retriever"},
        )

        result = await ainvoke_rag_app(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_retriever_langchain_schema.json")
    async def test_async_retrieve_langchain_docs(self):
        """Test async retrieval of LangChain-related documents."""
        callback = CallbackHandler(
            name="langchain-async-retriever-langchain",
            tags=["langchain", "async", "retriever", "langchain-docs"],
        )

        result = await ainvoke_rag_app(
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
# ASYNC AGENT TESTS
# =============================================================================


class TestAsyncAgentApp:
    """Async tests for agent-style LangChain app."""

    @pytest.mark.asyncio
    @trace_test("langchain_async_agent_simple_schema.json")
    async def test_async_simple_agent(self):
        """Test async simple agent with one tool call."""
        callback = CallbackHandler(
            name="langchain-async-agent-simple",
            tags=["langchain", "async", "agent", "simple"],
            metadata={"test_type": "async_agent"},
        )

        result = await ainvoke_simple_agent(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_agent_multi_step_schema.json")
    async def test_async_multi_step_agent(self):
        """Test async agent with multiple sequential tool calls."""
        callback = CallbackHandler(
            name="langchain-async-agent-multi-step",
            tags=["langchain", "async", "agent", "multi-step"],
        )

        result = await ainvoke_multi_step_agent(
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

    @pytest.mark.asyncio
    @trace_test("langchain_async_agent_complex_schema.json")
    async def test_async_complex_agent(self):
        """Test async agent with complex multi-tool workflow."""
        callback = CallbackHandler(
            name="langchain-async-agent-complex",
            tags=["langchain", "async", "agent", "complex"],
        )

        result = await ainvoke_complex_agent(
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
