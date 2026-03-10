import os
import pytest
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)
from deepeval.metrics import AnswerRelevancyMetric

from tests.test_integrations.test_agentcore.apps.agentcore_simple_app import (
    init_simple_agentcore,
    invoke_simple_agent,
)
from tests.test_integrations.test_agentcore.apps.agentcore_tool_app import (
    init_tool_agentcore,
    invoke_tool_agent,
)
from tests.test_integrations.test_agentcore.apps.agentcore_multiple_tools_app import (
    init_multiple_tools_agentcore,
    invoke_multiple_tools_agent,
)
from tests.test_integrations.test_agentcore.apps.agentcore_eval_app import (
    init_evals_agentcore,
    invoke_evals_agent,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("AWS_ACCESS_KEY_ID"),
    reason="AWS credentials are required to run Bedrock AgentCore tests.",
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

    @trace_test("agentcore_simple_schema.json")
    def test_simple_greeting(self):
        invoke_func = init_simple_agentcore(
            name="agentcore-simple-test",
            tags=["agentcore", "simple"],
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

    @trace_test("agentcore_tool_schema.json")
    def test_tool_calculation(self):
        invoke_func = init_tool_agentcore(
            name="agentcore-tool-test",
            tags=["agentcore", "tool"],
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

    @trace_test("agentcore_tool_metric_collection_schema.json")
    def test_tool_metric_collection(self):
        invoke_func = init_tool_agentcore(
            name="agentcore-tool-metric-test",
            tags=["agentcore", "tool", "metric-collection"],
            metadata={"test_type": "tool_metric_collection"},
            thread_id="tool-metric-123",
            user_id="test-user",
            tool_metric_collection_map={"calculate": "calculator-metrics"},
        )

        result = invoke_tool_agent(
            "What is 15 plus 25?",
            invoke_func=invoke_func,
        )

        assert result is not None
        assert "40" in result


class TestMultipleToolsApp:

    @trace_test("agentcore_multiple_tools_weather_schema.json")
    def test_multiple_tools_weather_only(self):
        invoke_func = init_multiple_tools_agentcore(
            name="agentcore-multiple-tools-weather",
            tags=["agentcore", "multiple-tools", "weather"],
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

    @trace_test("agentcore_multiple_tools_time_schema.json")
    def test_multiple_tools_time_only(self):
        invoke_func = init_multiple_tools_agentcore(
            name="agentcore-multiple-tools-time",
            tags=["agentcore", "multiple-tools", "time"],
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

    @trace_test("agentcore_parallel_tools_schema.json")
    def test_parallel_tool_calls(self):
        invoke_func = init_multiple_tools_agentcore(
            name="agentcore-parallel-tools",
            tags=["agentcore", "parallel-tools"],
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

    @trace_test("agentcore_features_sync.json")
    def test_full_features_sync(self):
        invoke_func = init_evals_agentcore(
            name="agentcore-full-features-sync",
            tags=["agentcore", "features", "sync"],
            metadata={"env": "testing", "priority": "high"},
            thread_id="thread-sync-features-001",
            user_id="user-sync-001",
            metric_collection="trace_metrics_v1",
            agent_metric_collection="agent_metrics_v1",
            llm_metric_collection="llm_metrics_v1",
            tool_metric_collection_map={"special_tool": "tool_metrics_v1"},
            trace_metric_collection="trace_metrics_override_v1",
            agent_metrics=[AnswerRelevancyMetric()],
        )

        result = invoke_evals_agent(
            "Use the special_tool to process 'Sync Data'",
            invoke_func=invoke_func,
        )

        assert result is not None
