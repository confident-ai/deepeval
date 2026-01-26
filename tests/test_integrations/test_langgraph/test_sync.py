"""
Sync LangGraph Tests
All synchronous tests using .invoke() and .stream()
"""

import os
from uuid import uuid4
from langchain_core.messages import HumanMessage
from deepeval.integrations.langchain import CallbackHandler
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

# App imports
from tests.test_integrations.test_langgraph.apps.langgraph_simple_app import (
    app as simple_app,
)
from tests.test_integrations.test_langgraph.apps.langgraph_multiple_tools_app import (
    app as multiple_tools_app,
)
from tests.test_integrations.test_langgraph.apps.langgraph_streaming_app import (
    sync_app as streaming_app,
)
from tests.test_integrations.test_langgraph.apps.langgraph_conditional_app import (
    app as conditional_app,
)
from tests.test_integrations.test_langgraph.apps.langgraph_parallel_tools_app import (
    sync_app as parallel_app,
)
from tests.test_integrations.test_langgraph.apps.langgraph_multi_turn_app import (
    get_app_with_memory,
    stateless_app,
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
# SIMPLE APP TESTS
# =============================================================================


class TestSimpleApp:
    """Tests for simple single-tool LangGraph agent."""

    @trace_test("langgraph_simple_schema.json")
    def test_weather_query(self):
        """Test a simple weather query that triggers one tool call."""
        callback = CallbackHandler(
            name="langgraph-simple-test",
            tags=["langgraph", "simple"],
            metadata={"test_type": "simple"},
            thread_id="simple-123",
            user_id="test-user",
        )

        result = simple_app.invoke(
            {
                "messages": [
                    HumanMessage(content="What's the weather in San Francisco?")
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0
        last_message = result["messages"][-1]
        assert hasattr(last_message, "content")


# # =============================================================================
# # MULTIPLE TOOLS TESTS
# # =============================================================================


class TestMultipleToolsApp:
    """Tests for multi-tool LangGraph agent."""

    @trace_test("langgraph_multiple_tools_schema.json")
    def test_city_info(self):
        """Test query that requires multiple tools about a city."""
        callback = CallbackHandler(
            name="langgraph-multi-tool-test",
            tags=["langgraph", "multiple-tools"],
            metadata={"test_type": "multiple_tools"},
            thread_id="multi-tool-123",
            user_id="test-user",
        )

        result = multiple_tools_app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Tell me about Tokyo - what's the weather, population, and timezone?"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @trace_test("langgraph_multiple_tools_mixed_schema.json")
    def test_mixed_query(self):
        """Test query that requires mixed tool types (info + calculation)."""
        callback = CallbackHandler(
            name="langgraph-mixed-tools-test",
            tags=["langgraph", "mixed-tools"],
            metadata={"test_type": "mixed_tools"},
        )

        result = multiple_tools_app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="What's the weather in Paris? Also calculate 100 * 1.5 + 50"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0


# =============================================================================
# STREAMING TESTS
# =============================================================================


class TestStreamingApp:
    """Tests for streaming LangGraph agent."""

    @trace_test("langgraph_streaming_schema.json")
    def test_sync_streaming(self):
        """Test sync streaming with tool calls."""
        callback = CallbackHandler(
            name="langgraph-streaming-sync",
            tags=["langgraph", "streaming", "sync"],
            metadata={"test_type": "streaming_sync"},
        )

        chunks = []
        for chunk in streaming_app.stream(
            {
                "messages": [
                    HumanMessage(content="What's the stock price of MSFT?")
                ]
            },
            config={"callbacks": [callback]},
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

    @trace_test("langgraph_streaming_multi_schema.json")
    def test_sync_streaming_multiple_tools(self):
        """Test sync streaming with multiple tool calls."""
        callback = CallbackHandler(
            name="langgraph-streaming-multi",
            tags=["langgraph", "streaming", "multi-tool"],
        )

        chunks = []
        for chunk in streaming_app.stream(
            {
                "messages": [
                    HumanMessage(
                        content="Get the stock price and company info for TSLA"
                    )
                ]
            },
            config={"callbacks": [callback]},
        ):
            chunks.append(chunk)

        assert len(chunks) > 0


# =============================================================================
# CONDITIONAL ROUTING TESTS
# =============================================================================


class TestConditionalApp:
    """Tests for conditional routing LangGraph agent."""

    @trace_test("langgraph_conditional_research_schema.json")
    def test_research_route(self):
        """Test routing to research node."""
        callback = CallbackHandler(
            name="langgraph-conditional-research",
            tags=["langgraph", "conditional", "research"],
            metadata={"test_type": "conditional_research"},
        )

        result = conditional_app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use the research tool exactly once to research: quantum computing. "
                            "Do not ask clarification questions. "
                            "After the tool returns, respond with a short 3-bullet summary and stop."
                        )
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @trace_test("langgraph_conditional_summarize_schema.json")
    def test_summarize_route(self):
        """Test routing to summarize node."""
        callback = CallbackHandler(
            name="langgraph-conditional-summarize",
            tags=["langgraph", "conditional", "summarize"],
        )

        result = conditional_app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Summarize this: Artificial intelligence is transforming industries worldwide."
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @trace_test("langgraph_conditional_fact_check_schema.json")
    def test_fact_check_route(self):
        """Test routing to fact check node."""
        callback = CallbackHandler(
            name="langgraph-conditional-factcheck",
            tags=["langgraph", "conditional", "fact-check"],
        )

        result = conditional_app.invoke(
            {
                "messages": [
                    HumanMessage(content="Fact check: The earth is round")
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @trace_test("langgraph_conditional_general_schema.json")
    def test_general_route(self):
        """Test routing to general node."""
        callback = CallbackHandler(
            name="langgraph-conditional-general",
            tags=["langgraph", "conditional", "general"],
        )

        result = conditional_app.invoke(
            {"messages": [HumanMessage(content="Hello, how are you today?")]},
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0


# =============================================================================
# PARALLEL TOOLS TESTS
# =============================================================================


class TestParallelToolsApp:
    """Tests for parallel tool execution LangGraph agent."""

    @trace_test("langgraph_parallel_weather_schema.json")
    def test_parallel_weather_queries(self):
        """Test parallel weather queries for multiple cities."""
        callback = CallbackHandler(
            name="langgraph-parallel-weather",
            tags=["langgraph", "parallel", "weather"],
            metadata={"test_type": "parallel_weather"},
        )

        result = parallel_app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="What's the weather in Tokyo, New York, and London?"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @trace_test("langgraph_parallel_mixed_schema.json")
    def test_parallel_mixed_tools(self):
        """Test parallel execution of different tool types."""
        callback = CallbackHandler(
            name="langgraph-parallel-mixed",
            tags=["langgraph", "parallel", "mixed"],
        )

        result = parallel_app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Get weather in Paris, stock price of TSLA, exchange rate USD to EUR, and calculate 100 * 1.5"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @trace_test("langgraph_parallel_stocks_schema.json")
    def test_parallel_stock_queries(self):
        """Test parallel stock price queries."""
        callback = CallbackHandler(
            name="langgraph-parallel-stocks",
            tags=["langgraph", "parallel", "stocks"],
        )

        result = parallel_app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Get stock prices for AAPL, GOOGL, MSFT, TSLA, and AMZN"
                    )
                ]
            },
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0


# =============================================================================
# MULTI-TURN TESTS
# =============================================================================


class TestMultiTurnApp:
    """Tests for multi-turn conversation LangGraph agent."""

    @trace_test("langgraph_multi_turn_schema.json")
    def test_multi_turn_shopping(self):
        """Test multi-turn shopping conversation with memory."""
        # Create fresh app instance to avoid state leakage between tests
        app = get_app_with_memory()
        thread_id = "test-shopping-001"

        # Turn 1: Add items
        callback1 = CallbackHandler(
            name="langgraph-multi-turn-1",
            tags=["langgraph", "multi-turn", "turn-1"],
            thread_id=thread_id,
            user_id="shopper-1",
        )
        result1 = app.invoke(
            {"messages": [HumanMessage(content="Add 3 apples to my cart")]},
            config={
                "callbacks": [callback1],
                "configurable": {"thread_id": thread_id},
            },
        )
        assert len(result1["messages"]) > 0

        # Turn 2: View cart
        callback2 = CallbackHandler(
            name="langgraph-multi-turn-2",
            tags=["langgraph", "multi-turn", "turn-2"],
            thread_id=thread_id,
            user_id="shopper-1",
        )
        result2 = app.invoke(
            {
                "messages": [
                    HumanMessage(content="Use view_cart to show what I have")
                ]
            },
            config={
                "callbacks": [callback2],
                "configurable": {"thread_id": thread_id},
            },
        )
        assert len(result2["messages"]) > 0

        # Turn 3: Apply coupon
        callback3 = CallbackHandler(
            name="langgraph-multi-turn-3",
            tags=["langgraph", "multi-turn", "turn-3"],
            thread_id=thread_id,
            user_id="shopper-1",
        )
        result3 = app.invoke(
            {"messages": [HumanMessage(content="Apply coupon SAVE10")]},
            config={
                "callbacks": [callback3],
                "configurable": {"thread_id": thread_id},
            },
        )
        assert len(result3["messages"]) > 0

    @trace_test("langgraph_stateless_schema.json")
    def test_stateless_single_turn(self):
        """Test single turn with stateless app."""
        callback = CallbackHandler(
            name="langgraph-stateless",
            tags=["langgraph", "stateless"],
        )

        result = stateless_app.invoke(
            {"messages": [HumanMessage(content="Add 3 oranges to my cart")]},
            config={"callbacks": [callback]},
        )

        assert len(result["messages"]) > 0

    @trace_test("langgraph_full_flow_schema.json")
    def test_full_shopping_flow(self):
        app = get_app_with_memory()

        # Prevent cross-run bleed from CallbackHandler’s class-level cache
        with CallbackHandler._thread_id_lock:
            CallbackHandler._thread_id_to_trace_uuid.clear()

        thread_id = f"full-flow-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        callback = CallbackHandler(
            name="langgraph-full-flow",
            tags=["langgraph", "full-flow"],
            thread_id=thread_id,
        )

        app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Add exactly 2 apples to the cart.\n"
                            "If you use tools in this system, you MUST call the tool required to update the cart.\n"
                            "Do not answer from memory."
                        )
                    )
                ]
            },
            config={**config, "callbacks": [callback]},
        )

        app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Apply the coupon code SAVE20.\n"
                            "You MUST call the coupon tool (do not apply it yourself).\n"
                            "Do not answer from memory."
                        )
                    )
                ]
            },
            config={**config, "callbacks": [callback]},
        )

        app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Proceed to checkout now.\n"
                            "You MUST call the checkout tool.\n"
                            "Do not answer from memory."
                        )
                    )
                ]
            },
            config={**config, "callbacks": [callback]},
        )

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Confirm my order.\n"
                            "You MUST call the confirm tool.\n"
                            "After tool output, reply with exactly: CONFIRMED"
                        )
                    )
                ]
            },
            config={**config, "callbacks": [callback]},
        )

        assert len(result["messages"]) > 0
