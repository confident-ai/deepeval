"""
Async LlamaIndex Tests
All asynchronous tests using .aquery(), .achat(), or .astream_chat()
"""

import os
import pytest
from deepeval.tracing import trace
from deepeval.prompt import Prompt

from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

from deepeval.tracing.trace_context import AgentSpanContext, LlmSpanContext
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.otel.test_exporter import test_exporter
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.tracing.context import current_trace_context, current_span_context
from tests.test_integrations.test_llamaindex.apps.eval_app import (
    get_evals_agent,
)

# App imports
from tests.test_integrations.test_llamaindex.apps.simple_app import (
    get_simple_engine,
)
from tests.test_integrations.test_llamaindex.apps.rag_app import get_rag_engine
from tests.test_integrations.test_llamaindex.apps.agent_app import get_agent
from tests.test_integrations.test_llamaindex.apps.router_app import (
    get_router_engine,
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"
prompt.label = "test-label"
prompt.hash = "bab04ec"


def trace_test(schema_name: str):
    """
    Decorator that switches between generate and assert mode based on GENERATE_SCHEMAS env var.
    """
    schema_path = os.path.join(_schemas_dir, schema_name)
    if is_generate_mode():
        os.makedirs(_schemas_dir, exist_ok=True)
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)


# =============================================================================
# ASYNC SIMPLE APP TESTS
# =============================================================================


class TestAsyncSimpleApp:
    """Tests for async LlamaIndex Query Engine."""

    @pytest.mark.asyncio
    @trace_test("llama_index_async_simple_schema.json")
    async def test_async_simple_query(self):
        """Test async basic query."""
        engine = get_simple_engine()
        with trace(
            name="llama_index_async_simple",
            tags=["llama_index", "async", "simple"],
            thread_id="llama_async_index_thread_id",
            user_id="llama_async_index_user_id",
            metrics=[AnswerRelevancyMetric()],
            metric_collection="llama_async_index_metric_collection",
        ):
            response = await engine.aquery("What is LlamaIndex?")
            assert "framework" in str(response).lower()


# =============================================================================
# ASYNC RAG APP TESTS
# =============================================================================


class TestAsyncRAGApp:
    """Tests for Async RAG."""

    @pytest.mark.asyncio
    @trace_test("llama_index_async_rag_schema.json")
    async def test_async_rag_query(self):
        """Test Async RAG retrieval."""
        engine = get_rag_engine()
        with trace(
            name="llama_index_async_rag",
            tags=["llama_index", "async", "rag"],
            thread_id="llama_async_index_thread_id",
            user_id="llama_async_index_user_id",
            metrics=[AnswerRelevancyMetric()],
            metric_collection="llama_async_index_metric_collection",
        ):
            response = await engine.aquery("What is Python?")
            assert "programming language" in str(response).lower()


# =============================================================================
# ASYNC AGENT APP TESTS
# =============================================================================


class TestAsyncAgentApp:
    """Tests for Async ReAct Agent."""

    @pytest.mark.asyncio
    @trace_test("llama_index_async_agent_schema.json")
    async def test_async_agent_tool(self):
        """Test Async Agent with tools."""
        agent = get_agent()
        with trace(
            name="llama_index_async_agent",
            tags=["llama_index", "async", "agent"],
            thread_id="llama_async_index_thread_id",
            user_id="llama_async_index_user_id",
            metrics=[AnswerRelevancyMetric()],
            metric_collection="llama_async_index_metric_collection",
        ):
            # For Workflow agents, use .run()
            response = await agent.run("What is the weather in Tokyo?")
            assert "cloudy" in str(response).lower()

    @pytest.mark.asyncio
    @trace_test("llama_index_async_agent_math_schema.json")
    async def test_async_agent_math(self):
        """Test Async Agent with math tool."""
        agent = get_agent()
        with trace(
            name="llama_index_async_agent",
            tags=["llama_index", "async", "agent", "math"],
            thread_id="llama_async_index_thread_id",
            user_id="llama_async_index_user_id",
            metrics=[AnswerRelevancyMetric()],
            metric_collection="llama_async_index_metric_collection",
        ):
            response = await agent.run("Calculate 50 * 2")
            assert "100" in str(response)


# =============================================================================
# ASYNC ROUTER APP TESTS
# =============================================================================


class TestAsyncRouterApp:
    """Tests for Async Routing."""

    @pytest.mark.asyncio
    @trace_test("llama_index_async_router_schema.json")
    async def test_async_router_selection(self):
        """Test Async Router selection."""
        engine = get_router_engine()
        with trace(
            name="llama_index_async_router",
            tags=["llama_index", "async", "router"],
            thread_id="llama_async_index_thread_id",
            user_id="llama_async_index_user_id",
            metrics=[AnswerRelevancyMetric()],
            metric_collection="llama_async_index_metric_collection",
        ):
            response = await engine.aquery("Calculate 21 + 21")
            assert "42" in str(response)


# =============================================================================
# DEEPEVAL FEATURES TESTS (ASYNC)
# =============================================================================


class TestDeepEvalFeaturesAsync:
    """Tests for DeepEval specific features based on official docs."""

    @pytest.fixture(autouse=True)
    def reset_instrumentation(self):
        """Reset ALL tracing state before each test."""
        trace_manager.clear_traces()
        test_exporter.clear_span_json_list()
        trace_testing_manager.test_dict = None
        current_trace_context.set(None)
        current_span_context.set(None)
        yield

    @pytest.mark.asyncio
    @trace_test("llama_index_features_async.json")
    async def test_features_async(self):
        """Test passing metric_collection and metadata in Async context."""
        agent = get_evals_agent()

        agent_ctx = AgentSpanContext(
            metric_collection="production_agent_metrics",
            metrics=[AnswerRelevancyMetric()],
            expected_output="exp output agent level async",
            context=["context here agent level async"],
        )
        llm_ctx = LlmSpanContext(
            metric_collection="production_llm_metrics",
            prompt=prompt,
            metrics=[AnswerRelevancyMetric()],
            expected_output="exp output llm level async",
            context=["context here llm level async"],
        )

        with trace(
            name="Calculation Check Async",
            tags=["production", "async"],
            metrics=[AnswerRelevancyMetric()],
            metric_collection="llama_async_index_metric_collection",
            user_id="user_async_456",
            thread_id="thread_async_XYZ",
            agent_span_context=agent_ctx,
            llm_span_context=llm_ctx,
        ):
            response = await agent.run("What is 4 * 6?")
            return response
