import pytest

from deepeval.metrics import ToolPermissionMetric
from deepeval.test_case import LLMTestCase, ToolCall


def _test_case(tool_names):
    return LLMTestCase(
        input="do the task",
        actual_output="done",
        tools_called=[ToolCall(name=name) for name in tool_names],
    )


class TestToolPermissionMetric:
    """ToolPermissionMetric is deterministic, so these run without any API key."""

    def test_all_calls_authorized_passes(self):
        metric = ToolPermissionMetric(allowed_tools=["search_kb", "reply"])
        metric.measure(_test_case(["search_kb", "reply"]))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_unauthorized_tool_fails(self):
        metric = ToolPermissionMetric(allowed_tools=["search_kb"])
        metric.measure(_test_case(["search_kb", "delete_account"]))
        assert metric.score == 0.5
        assert metric.is_successful() is False
        assert "delete_account" in metric.reason

    def test_denied_tool_fails_even_if_allowed(self):
        metric = ToolPermissionMetric(
            allowed_tools=["search_kb", "wire_transfer"],
            denied_tools=["wire_transfer"],
        )
        metric.measure(_test_case(["wire_transfer"]))
        assert metric.score == 0.0
        assert metric.is_successful() is False

    def test_no_tools_called_passes(self):
        metric = ToolPermissionMetric(allowed_tools=["search_kb"])
        metric.measure(_test_case([]))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_denylist_only(self):
        metric = ToolPermissionMetric(denied_tools=["rm_rf"])
        metric.measure(_test_case(["safe_tool", "rm_rf"]))
        assert metric.score == 0.5
        assert metric.is_successful() is False

    def test_partial_credit_with_threshold(self):
        # 2 of 3 authorized -> ~0.67; passes at threshold 0.6.
        metric = ToolPermissionMetric(allowed_tools=["a", "b"], threshold=0.6)
        metric.measure(_test_case(["a", "b", "c"]))
        assert round(metric.score, 2) == 0.67
        assert metric.is_successful() is True

    def test_strict_mode_zeroes_partial_success(self):
        metric = ToolPermissionMetric(allowed_tools=["a"], strict_mode=True)
        metric.measure(_test_case(["a", "b"]))
        assert metric.score == 0
        assert metric.is_successful() is False

    def test_requires_a_policy(self):
        with pytest.raises(ValueError):
            ToolPermissionMetric()

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = ToolPermissionMetric(allowed_tools=["a"])
        score = await metric.a_measure(_test_case(["a", "b"]))
        assert score == 0.5
        assert metric.is_successful() is False
