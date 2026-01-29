"""
Sync LlamaIndex Tests
All synchronous tests using .query(), .chat(), or .stream_chat()

NOTE: Run with GENERATE_SCHEMAS=1 first to generate the JSON schemas.
"""

import os
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
# SIMPLE APP TESTS
# =============================================================================


class TestSimpleApp:
    """Tests for basic LlamaIndex Query Engine."""

    @trace_test("llama_index_simple_schema.json")
    def test_simple_query(self):
        """Test a basic query without tools or complex retrieval."""
        engine = get_simple_engine()
        with trace(
            name="llama_index_simple",
            tags=["llama_index", "simple"],
        ):
            response = engine.query("What is LlamaIndex?")
            assert "framework" in str(response).lower()


# =============================================================================
# RAG APP TESTS
# =============================================================================


class TestRAGApp:
    """Tests for Retrieval-Augmented Generation."""

    @trace_test("llama_index_rag_python_schema.json")
    def test_rag_python_query(self):
        """Test RAG retrieval for 'Python' keyword."""
        engine = get_rag_engine()
        with trace(
            name="llama_index_rag",
            tags=["llama_index", "rag", "python"],
        ):
            response = engine.query("What is Python?")
            assert "programming language" in str(response).lower()

    @trace_test("llama_index_rag_llama_schema.json")
    def test_rag_llama_query(self):
        """Test RAG retrieval for 'LlamaIndex' keyword."""
        engine = get_rag_engine()
        with trace(
            name="llama_index_rag",
            tags=["llama_index", "rag", "llama"],
        ):
            response = engine.query("What is LlamaIndex?")
            assert "data framework" in str(response).lower()


# =============================================================================
# ROUTER APP TESTS
# =============================================================================


class TestRouterApp:
    """Tests for Router Query Engine."""

    @trace_test("llama_index_router_math_schema.json")
    def test_router_math_selection(self):
        """Test Router correctly selecting the Math engine."""
        engine = get_router_engine()
        with trace(
            name="llama_index_router",
            tags=["llama_index", "router"],
        ):
            # This query should route to the MockMathEngine
            response = engine.query("Calculate 21 + 21")
            assert "42" in str(response)
