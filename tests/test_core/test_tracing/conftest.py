"""
Shared test utilities for tracing tests.

This module:
- Imports and re-exports schema validation utilities from test_integrations/utils.py
- Provides tracing-specific fixtures
- Provides helper function for creating trace_test decorator with schema paths
"""

import os
import asyncio
import pytest

# Re-export utilities from test_integrations/utils.py
from tests.test_integrations.utils import (
    assert_json_object_structure,
    load_trace_data,
    generate_trace_json,
    assert_trace_json,
)

# Configuration for generate vs assert mode
# Set to True to generate schemas, False to assert against existing schemas
# Can be overridden via environment variable: GENERATE_SCHEMAS=true pytest ...
GENERATE_MODE = os.environ.get("GENERATE_SCHEMAS", "").lower() in (
    "true",
    "1",
    "yes",
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")


def get_schema_path(schema_name: str) -> str:
    """Get the full path to a schema file relative to the test_tracing/schemas directory."""
    return os.path.join(_schemas_dir, schema_name)


def trace_test(schema_name: str):
    """
    Decorator that switches between generate and assert mode based on GENERATE_MODE.

    Args:
        schema_name: Name of the schema file (relative path from schemas/ directory)
    """
    schema_path = get_schema_path(schema_name)
    if GENERATE_MODE:
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)


# Common fixtures
@pytest.fixture(autouse=True)
def ensure_event_loop():
    """Ensure an event loop exists for sync tests that need async operations."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield
    # Don't close the loop - other tests may need it


@pytest.fixture(autouse=True)
def silence_confident_trace(monkeypatch):
    """Silence trace posting during tests."""
    from deepeval.tracing.tracing import trace_manager

    monkeypatch.setenv("CONFIDENT_TRACE_FLUSH", "0")
    monkeypatch.setattr(
        trace_manager, "post_trace", lambda *a, **k: None, raising=True
    )


@pytest.fixture(autouse=True)
def reset_trace_state():
    """Reset trace manager state before and after each test."""
    from deepeval.tracing.tracing import trace_manager
    from deepeval.tracing.context import (
        current_trace_context,
        current_span_context,
    )

    # Reset BEFORE each test to ensure clean state
    current_trace_context.set(None)
    current_span_context.set(None)
    trace_manager.clear_traces()

    yield

    # Reset AFTER each test
    current_trace_context.set(None)
    current_span_context.set(None)
    trace_manager.traces_to_evaluate_order.clear()
    trace_manager.traces_to_evaluate.clear()
    trace_manager.integration_traces_to_evaluate.clear()
    trace_manager.test_case_metrics.clear()
    trace_manager.trace_uuid_to_golden.clear()
    trace_manager.clear_traces()


def get_active_trace_and_span():
    """Helper to peek at current trace/span via the observer context."""
    from deepeval.tracing.context import (
        current_trace_context,
        current_span_context,
    )

    return current_trace_context.get(), current_span_context.get()
