import os

import pytest

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing import next_agent_span, next_llm_span, next_tool_span
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

from tests.test_integrations.test_strands.apps.strands_simple_app import (
    init_simple_strands,
    invoke_simple_agent,
)
from tests.test_integrations.test_strands.apps.strands_tool_app import (
    init_tool_strands,
    invoke_tool_agent,
)
from tests.test_integrations.test_strands.apps.strands_multiple_tools_app import (
    init_multiple_tools_strands,
    invoke_multiple_tools_agent,
)
from tests.test_integrations.test_strands.apps.strands_eval_app import (
    init_evals_strands,
    invoke_evals_agent,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required to run Strands integration tests "
    "(the OpenAIModel provider proxies to OpenAI's API).",
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")


def trace_test(schema_name: str):
    schema_path = os.path.join(_schemas_dir, schema_name)
    if is_generate_mode():
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)


class TestSimpleApp:

    @trace_test("strands_simple_schema.json")
    def test_simple_greeting(self):
        invoke_func = init_simple_strands(
            name="strands-simple-test",
            tags=["strands", "simple"],
            metadata={"test_type": "simple"},
            thread_id="simple-123",
            user_id="test-user",
        )

        result = invoke_simple_agent(
            "Say hello in exactly three words.",
            invoke_func=invoke_func,
        )

        assert result is not None
        assert len(result) > 0


class TestToolApp:

    @trace_test("strands_tool_schema.json")
    def test_tool_calculation(self):
        invoke_func = init_tool_strands(
            name="strands-tool-test",
            tags=["strands", "tool"],
            metadata={"test_type": "tool"},
            thread_id="tool-123",
            user_id="test-user",
        )

        result = invoke_tool_agent(
            "What is 7 multiplied by 8?",
            invoke_func=invoke_func,
        )

        assert result is not None
        assert "56" in result

    @trace_test("strands_tool_metric_collection_schema.json")
    def test_tool_metric_collection(self):
        """Tool-level metric_collection now flows through
        ``with next_tool_span(metric_collection=...)`` at the call
        site instead of a top-level ``tool_metric_collection_map``
        kwarg on ``instrument_strands``.

        ``next_tool_span`` is one-shot — it hits the FIRST tool span
        emitted inside the ``with`` block, which matches the
        single-tool-call test below."""
        invoke_func = init_tool_strands(
            name="strands-tool-metric-test",
            tags=["strands", "tool", "metric-collection"],
            metadata={"test_type": "tool_metric_collection"},
            thread_id="tool-metric-123",
            user_id="test-user",
        )

        with next_tool_span(metric_collection="calculator-metrics"):
            result = invoke_tool_agent(
                "What is 15 plus 25?",
                invoke_func=invoke_func,
            )

        assert result is not None
        assert "40" in result


class TestMultipleToolsApp:

    @trace_test("strands_multiple_tools_weather_schema.json")
    def test_multiple_tools_weather_only(self):
        invoke_func = init_multiple_tools_strands(
            name="strands-multiple-tools-weather",
            tags=["strands", "multiple-tools", "weather"],
            metadata={"test_type": "multiple_tools_weather"},
            thread_id="multiple-tools-weather-123",
            user_id="test-user",
        )

        result = invoke_multiple_tools_agent(
            "Use the get_weather tool exactly once to get the weather in Tokyo.",
            invoke_func=invoke_func,
        )

        assert result is not None
        assert "72" in result or "sunny" in result.lower()

    @trace_test("strands_multiple_tools_time_schema.json")
    def test_multiple_tools_time_only(self):
        invoke_func = init_multiple_tools_strands(
            name="strands-multiple-tools-time",
            tags=["strands", "multiple-tools", "time"],
            metadata={"test_type": "multiple_tools_time"},
            thread_id="multiple-tools-time-123",
            user_id="test-user",
        )

        result = invoke_multiple_tools_agent(
            "Use the get_time tool exactly once to get the current time in London.",
            invoke_func=invoke_func,
        )

        assert result is not None
        assert "7:00" in result or "GMT" in result

    @trace_test("strands_parallel_tools_schema.json")
    def test_parallel_tool_calls(self):
        invoke_func = init_multiple_tools_strands(
            name="strands-parallel-tools",
            tags=["strands", "parallel-tools"],
            metadata={"test_type": "parallel_tools"},
            thread_id="parallel-tools-123",
            user_id="test-user",
        )

        result = invoke_multiple_tools_agent(
            "Use both the get_weather tool AND the get_time tool for Paris. "
            "Call both tools exactly once each.",
            invoke_func=invoke_func,
        )

        assert result is not None
        assert "62" in result or "cloudy" in result.lower()
        assert "8:00" in result or "CET" in result


class TestDeepEvalFeatures:
    """Span-level configuration migrates to per-call ``with next_*_span(...)``.

    Mirrors ``test_agentcore.test_sync.TestDeepEvalFeatures``: stacked
    ``with`` blocks stage values for the next agent / LLM / tool span
    emitted inside the wrapper. The ``special_tool`` itself uses
    ``update_current_span(...)`` from inside its body for its own
    metric collection — handled in ``apps/strands_eval_app.py``."""

    @trace_test("strands_features_sync.json")
    def test_full_features_sync(self):
        invoke_func = init_evals_strands(
            name="strands-full-features-sync",
            tags=["strands", "features", "sync"],
            metadata={"env": "testing", "priority": "high"},
            thread_id="thread-sync-features-001",
            user_id="user-sync-001",
            metric_collection="trace_metrics_override_v1",
        )

        with next_agent_span(
            metric_collection="agent_metrics_v1",
            metrics=[AnswerRelevancyMetric()],
        ), next_llm_span(metric_collection="llm_metrics_v1"):
            result = invoke_evals_agent(
                "Use the special_tool to process 'Sync Data'",
                invoke_func=invoke_func,
            )

        assert result is not None
