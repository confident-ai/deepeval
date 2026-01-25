"""
Async LangGraph Tests
All asynchronous tests using .ainvoke() and .astream()
"""

import os
import pytest
from langchain_core.messages import HumanMessage
from deepeval.integrations.langchain import CallbackHandler
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
)

# App imports
from tests.test_integrations.test_langgraph.apps.langgraph_async_app import (
    app as async_app,
)
from tests.test_integrations.test_langgraph.apps.langgraph_streaming_app import (
    async_app as streaming_async_app,
)
from tests.test_integrations.test_langgraph.apps.langgraph_conditional_app import (
    app as conditional_app,
)
from tests.test_integrations.test_langgraph.apps.langgraph_parallel_tools_app import (
    async_app as parallel_async_app,
)
from tests.test_integrations.test_langgraph.apps.langgraph_multi_turn_app import (
    get_async_app_with_memory,
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
# ASYNC APP TESTS
# =============================================================================


class TestAsyncApp:
    """Tests for async LangGraph agent invocation."""

    @pytest.mark.asyncio
    @trace_test("langgraph_async_single_tool_schema.json")
    async def test_single_tool(self):
        """Test async invocation with a single tool call."""
        callback = CallbackHandler(
            name="langgraph-async-single",
            tags=["langgraph", "async", "single-tool"],
            metadata={"test_type": "async_single"},
            thread_id="async-single-123",
            user_id="async-user",
        )

        result = await async_app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use the search_database tool to look up 'Rust (programming language)'. "
                            "Do not ask clarification questions."
                        )
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    @trace_test("langgraph_async_multiple_tools_schema.json")
    async def test_multiple_tools(self):
        """Test async invocation with multiple tool calls."""
        callback = CallbackHandler(
            name="langgraph-async-multi",
            tags=["langgraph", "async", "multi-tool"],
            metadata={"test_type": "async_multi"},
            thread_id="async-multi-123",
            user_id="async-user",
        )

        result = await async_app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use the search_database tool to look up 'Python (programming language)'. "
                            "Then translate the result to Spanish using the translate tool. "
                            "Do not ask clarification questions."
                        )
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    @trace_test("langgraph_async_no_tools_schema.json")
    async def test_no_tool_needed(self):
        """Test async invocation where no tool is needed."""
        callback = CallbackHandler(
            name="langgraph-async-no-tools",
            tags=["langgraph", "async", "no-tools"],
        )

        result = await async_app.ainvoke(
            {"messages": [HumanMessage(content="Hello, how are you?")]},
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0


# =============================================================================
# ASYNC STREAMING TESTS
# =============================================================================


class TestAsyncStreamingApp:
    """Tests for async streaming LangGraph agent."""

    @pytest.mark.asyncio
    @trace_test("langgraph_async_streaming_schema.json")
    async def test_async_streaming(self):
        """Test async streaming with tool calls."""
        callback = CallbackHandler(
            name="langgraph-streaming-async",
            tags=["langgraph", "streaming", "async"],
            metadata={"test_type": "streaming_async"},
        )

        chunks = []
        async for chunk in streaming_async_app.astream(
            {
                "messages": [
                    HumanMessage(content="What's the stock price of GOOGL?")
                ]
            },
            config={"callbacks": [callback]},
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    @trace_test("langgraph_async_streaming_multi_schema.json")
    async def test_async_streaming_multiple_tools(self):
        """Test async streaming with multiple tool calls."""
        callback = CallbackHandler(
            name="langgraph-streaming-async-multi",
            tags=["langgraph", "streaming", "async", "multi"],
        )

        chunks = []
        async for chunk in streaming_async_app.astream(
            {
                "messages": [
                    HumanMessage(
                        content="Get the stock price and company info for AMZN"
                    )
                ]
            },
            config={"callbacks": [callback]},
        ):
            chunks.append(chunk)

        assert len(chunks) > 0


# =============================================================================
# ASYNC CONDITIONAL ROUTING TESTS
# =============================================================================


class TestAsyncConditionalApp:
    """Tests for async conditional routing LangGraph agent."""

    @pytest.mark.asyncio
    @trace_test("langgraph_async_conditional_schema.json")
    async def test_async_conditional_routing(self):
        """Test async conditional routing."""
        callback = CallbackHandler(
            name="langgraph-conditional-async",
            tags=["langgraph", "conditional", "async"],
        )

        result = await conditional_app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use the research tool to research: latest developments in space exploration. "
                            "Do not ask clarification questions. "
                            "Return a short summary of the findings."
                        )
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0


# =============================================================================
# ASYNC PARALLEL TOOLS TESTS
# =============================================================================


class TestAsyncParallelToolsApp:
    """Tests for async parallel tool execution LangGraph agent."""

    @pytest.mark.asyncio
    @trace_test("langgraph_async_parallel_schema.json")
    async def test_async_parallel_tools(self):
        """Test async parallel tool execution."""
        callback = CallbackHandler(
            name="langgraph-parallel-async",
            tags=["langgraph", "parallel", "async"],
        )

        result = await parallel_async_app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Do the following using tools (do not ask clarification questions):"
                            "1) Call get_weather with location=Sydney, Australia. "
                            "2) Call get_weather with location=Tokyo, Japan. "
                            "3) Call search_news with topic=tech. "
                            "Then return a short combined result."
                        )
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    @trace_test("langgraph_async_parallel_heavy_schema.json")
    async def test_async_heavy_parallel(self):
        """Test async with many parallel tool calls."""
        callback = CallbackHandler(
            name="langgraph-parallel-async-heavy",
            tags=["langgraph", "parallel", "async", "heavy"],
        )

        result = await parallel_async_app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="""
                            Do the following using tools (do not ask clarification questions).
                            Call each tool exactly once per bullet item, using the exact parameters shown.

                            1) get_weather:
                               - location="Tokyo, Japan"
                               - location="New York, NY"
                               - location="London, UK"
                               - location="Paris, France"
                               - location="Sydney, Australia"

                            2) finance (stock prices):
                               - ticker="AAPL", type="equity", market="USA"
                               - ticker="GOOGL", type="equity", market="USA"
                               - ticker="MSFT", type="equity", market="USA"

                            3) calculator (exchange rates and percentage math):
                               - expression="USD to EUR exchange rate"
                               - expression="USD to GBP exchange rate"
                               - expression="0.15*378.90"

                            Then return a short report with the results.
                        """
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0


# =============================================================================
# ASYNC MULTI-TURN TESTS
# =============================================================================


class TestAsyncMultiTurnApp:
    """Tests for async multi-turn conversation LangGraph agent."""

    @pytest.mark.asyncio
    @trace_test("langgraph_async_multi_turn_schema.json")
    async def test_async_multi_turn(self):
        """Test async multi-turn conversation."""
        # Create fresh app instance to avoid state leakage between tests
        app = get_async_app_with_memory()
        thread_id = "async-shopping-001"

        # Turn 1
        callback1 = CallbackHandler(
            name="langgraph-async-multi-1",
            tags=["langgraph", "async", "multi-turn"],
            thread_id=thread_id,
        )
        result1 = await app.ainvoke(
            {"messages": [HumanMessage(content="Add 5 apples to cart")]},
            config={
                "callbacks": [callback1],
                "configurable": {"thread_id": thread_id},
            },
        )
        assert len(result1["messages"]) > 0

        # Turn 2
        callback2 = CallbackHandler(
            name="langgraph-async-multi-2",
            tags=["langgraph", "async", "multi-turn"],
            thread_id=thread_id,
        )
        result2 = await app.ainvoke(
            {"messages": [HumanMessage(content="Apply FREESHIP coupon")]},
            config={
                "callbacks": [callback2],
                "configurable": {"thread_id": thread_id},
            },
        )
        assert len(result2["messages"]) > 0
