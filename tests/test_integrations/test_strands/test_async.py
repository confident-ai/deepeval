import os

import pytest

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing import next_agent_span, next_llm_span
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

from tests.test_integrations.test_strands.apps.strands_simple_app import (
    init_simple_strands,
    ainvoke_simple_agent,
)
from tests.test_integrations.test_strands.apps.strands_tool_app import (
    init_tool_strands,
    ainvoke_tool_agent,
)
from tests.test_integrations.test_strands.apps.strands_multiple_tools_app import (
    init_multiple_tools_strands,
    ainvoke_multiple_tools_agent,
)
from tests.test_integrations.test_strands.apps.strands_eval_app import (
    init_evals_strands,
    ainvoke_evals_agent,
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


class TestAsyncSimpleApp:

    @pytest.mark.asyncio
    @trace_test("strands_async_simple_schema.json")
    async def test_async_simple_greeting(self):
        invoke_func = init_simple_strands(
            name="strands-async-simple-test",
            tags=["strands", "simple", "async"],
            metadata={"test_type": "async_simple"},
            thread_id="async-simple-123",
            user_id="test-user-async",
        )

        result = await ainvoke_simple_agent(
            "Say hello in exactly three words.",
            invoke_func=invoke_func,
        )

        assert result is not None
        assert len(result) > 0


class TestAsyncToolApp:

    @pytest.mark.asyncio
    @trace_test("strands_async_tool_schema.json")
    async def test_async_tool_calculation(self):
        invoke_func = init_tool_strands(
            name="strands-async-tool-test",
            tags=["strands", "tool", "async"],
            metadata={"test_type": "async_tool"},
            thread_id="async-tool-123",
            user_id="test-user-async",
        )

        result = await ainvoke_tool_agent(
            "What is 9 multiplied by 6?",
            invoke_func=invoke_func,
        )

        assert result is not None
        assert "54" in result


class TestAsyncMultipleToolsApp:

    @pytest.mark.asyncio
    @trace_test("strands_async_parallel_tools_schema.json")
    async def test_async_parallel_tool_calls(self):
        invoke_func = init_multiple_tools_strands(
            name="strands-async-parallel-tools",
            tags=["strands", "parallel-tools", "async"],
            metadata={"test_type": "async_parallel_tools"},
            thread_id="async-parallel-tools-123",
            user_id="test-user-async",
        )

        result = await ainvoke_multiple_tools_agent(
            "Use both the get_weather tool AND the get_time tool for Tokyo. "
            "Call both tools exactly once each.",
            invoke_func=invoke_func,
        )

        assert result is not None
        assert "72" in result or "sunny" in result.lower()
        assert "3:00" in result or "JST" in result


class TestDeepEvalFeaturesAsync:
    """Async equivalent of ``TestDeepEvalFeatures``: span-level kwargs
    migrate from ``init_evals_strands(...)`` to per-call
    ``with next_*_span(...)`` blocks. The ``special_tool`` itself
    sets its own ``metric_collection`` via ``update_current_span(...)``
    — see ``apps/strands_eval_app.py``."""

    @pytest.mark.asyncio
    @trace_test("strands_features_async.json")
    async def test_full_features_async(self):
        invoke_func = init_evals_strands(
            name="strands-full-features-async",
            tags=["strands", "features", "async"],
            metadata={"env": "testing_async", "mode": "async"},
            thread_id="thread-async-features-002",
            user_id="user-async-002",
            metric_collection="trace_metrics_override_async_v1",
        )

        with next_agent_span(
            metric_collection="agent_metrics_async_v1",
            metrics=[AnswerRelevancyMetric()],
        ), next_llm_span(metric_collection="llm_metrics_async_v1"):
            result = await ainvoke_evals_agent(
                "Use the special_tool to process 'Async Data'",
                invoke_func=invoke_func,
            )

        assert result is not None
