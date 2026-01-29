"""
Async LlamaIndex Tests
All asynchronous tests using .aquery(), .achat(), or .astream_chat()
"""

import os
import pytest
from deepeval.tracing import trace

from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
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
        ):
            response = await engine.aquery("Calculate 21 + 21")
            assert "42" in str(response)
