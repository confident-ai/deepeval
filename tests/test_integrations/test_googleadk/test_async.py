"""Asynchronous end-to-end traces for the Google ADK integration.

Mirrors the AgentCore ``test_async.py`` class layout: ``TestAsyncSimpleApp``,
``TestAsyncToolApp``, ``TestAsyncMultipleToolsApp``,
``TestDeepEvalFeaturesAsync``. Drives the agent through
``runner.run_async(...)`` so the OpenInference instrumentor's
async-path span emission is exercised.

Schema regeneration: ``GENERATE_SCHEMAS=true pytest tests/test_integrations/test_googleadk/test_async.py``.
See ``schemas/README.md``.

Skipped without ``GOOGLE_API_KEY``.
"""

import os

import pytest

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing import next_agent_span, next_llm_span

from tests.test_integrations.test_googleadk.apps.googleadk_simple_app import (
    init_simple_googleadk,
    ainvoke_simple_agent,
)
from tests.test_integrations.test_googleadk.apps.googleadk_tool_app import (
    init_tool_googleadk,
    ainvoke_tool_agent,
)
from tests.test_integrations.test_googleadk.apps.googleadk_multiple_tools_app import (
    init_multiple_tools_googleadk,
    ainvoke_multiple_tools_agent,
)
from tests.test_integrations.test_googleadk.apps.googleadk_eval_app import (
    init_evals_googleadk,
    ainvoke_evals_agent,
)
from tests.test_integrations.test_googleadk.conftest import trace_test


pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY is required to run Google ADK tests against Gemini.",
)


class TestAsyncSimpleApp:

    @pytest.mark.asyncio
    @trace_test("googleadk_async_simple_schema.json")
    async def test_async_simple_greeting(self):
        invoke_func = init_simple_googleadk(
            name="googleadk-async-simple-test",
            tags=["googleadk", "simple", "async"],
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
    @trace_test("googleadk_async_tool_schema.json")
    async def test_async_tool_calculation(self):
        invoke_func = init_tool_googleadk(
            name="googleadk-async-tool-test",
            tags=["googleadk", "tool", "async"],
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
    @trace_test("googleadk_async_parallel_tools_schema.json")
    async def test_async_parallel_tool_calls(self):
        invoke_func = init_multiple_tools_googleadk(
            name="googleadk-async-parallel-tools",
            tags=["googleadk", "parallel-tools", "async"],
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
    migrate from ``init_evals_googleadk(...)`` to per-call
    ``with next_*_span(...)`` blocks. The ``special_tool`` itself
    sets its own ``metric_collection`` via ``update_current_span(...)``
    — see ``apps/googleadk_eval_app.py``."""

    @pytest.mark.asyncio
    @trace_test("googleadk_features_async.json")
    async def test_full_features_async(self):
        invoke_func = init_evals_googleadk(
            name="googleadk-full-features-async",
            tags=["googleadk", "features", "async"],
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
