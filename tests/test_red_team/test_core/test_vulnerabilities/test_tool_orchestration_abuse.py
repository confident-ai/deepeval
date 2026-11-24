import pytest

from deepteam.vulnerabilities import ToolOrchestrationAbuse
from deepteam.vulnerabilities.tool_orchestration_abuse import (
    ToolOrchestrationAbuseType,
)
from deepteam.test_case import RTTestCase


class TestToolOrchestrationAbuse:

    def test_tool_orchestration_abuse_all_types(self):
        types = [
            "recursive_tool_calls",
            "unsafe_tool_composition",
            "tool_budget_exhaustion",
            "cross_tool_state_leakage",
        ]
        tool_orchestration_abuse = ToolOrchestrationAbuse(types=types)
        assert sorted(
            type.value for type in tool_orchestration_abuse.types
        ) == sorted(types)

    def test_tool_orchestration_abuse_all_types_default(self):
        tool_orchestration_abuse = ToolOrchestrationAbuse()
        assert sorted(
            type.value for type in tool_orchestration_abuse.types
        ) == sorted(type.value for type in ToolOrchestrationAbuseType)

    def test_tool_orchestration_abuse_recursive_tool_calls(self):
        types = ["recursive_tool_calls"]
        tool_orchestration_abuse = ToolOrchestrationAbuse(types=types)
        assert sorted(
            type.value for type in tool_orchestration_abuse.types
        ) == sorted(types)

    def test_tool_orchestration_abuse_unsafe_tool_composition(self):
        types = ["unsafe_tool_composition"]
        tool_orchestration_abuse = ToolOrchestrationAbuse(types=types)
        assert sorted(
            type.value for type in tool_orchestration_abuse.types
        ) == sorted(types)

    def test_tool_orchestration_abuse_tool_budget_exhaustion(self):
        types = ["tool_budget_exhaustion"]
        tool_orchestration_abuse = ToolOrchestrationAbuse(types=types)
        assert sorted(
            type.value for type in tool_orchestration_abuse.types
        ) == sorted(types)

    def test_tool_orchestration_abuse_cross_tool_state_leakage(self):
        types = ["cross_tool_state_leakage"]
        tool_orchestration_abuse = ToolOrchestrationAbuse(types=types)
        assert sorted(
            type.value for type in tool_orchestration_abuse.types
        ) == sorted(types)

    def test_tool_orchestration_abuse_all_types_invalid(self):
        types = [
            "recursive_tool_calls",
            "unsafe_tool_composition",
            "tool_budget_exhaustion",
            "cross_tool_state_leakage",
            "invalid",
        ]
        with pytest.raises(ValueError):
            ToolOrchestrationAbuse(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        tool_orchestration_abuse = ToolOrchestrationAbuse(
            types=["recursive_tool_calls"]
        )
        test_cases = tool_orchestration_abuse.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Tool Orchestration Abuse" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type
            == ToolOrchestrationAbuseType.RECURSIVE_TOOL_CALLS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        tool_orchestration_abuse = ToolOrchestrationAbuse(
            types=["unsafe_tool_composition"], async_mode=False
        )

        def dummy_model_callback(prompt):
            return prompt

        results = tool_orchestration_abuse.assess(
            model_callback=dummy_model_callback,
        )

        assert tool_orchestration_abuse.is_vulnerable() is not None
        assert (
            tool_orchestration_abuse.simulated_attacks is not None
            and isinstance(tool_orchestration_abuse.simulated_attacks, dict)
        )
        assert tool_orchestration_abuse.res is not None and isinstance(
            tool_orchestration_abuse.res, dict
        )
        assert ToolOrchestrationAbuseType.UNSAFE_TOOL_COMPOSITION in results
        assert (
            len(results[ToolOrchestrationAbuseType.UNSAFE_TOOL_COMPOSITION])
            == 1
        )
        test_case = results[ToolOrchestrationAbuseType.UNSAFE_TOOL_COMPOSITION][
            0
        ]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_tool_orchestration_metric(self):
        from deepteam.metrics import ToolOrchestrationMetric

        tool_orchestration_abuse = ToolOrchestrationAbuse(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = tool_orchestration_abuse._get_metric(
            ToolOrchestrationAbuseType.UNSAFE_TOOL_COMPOSITION
        )
        assert isinstance(metric, ToolOrchestrationMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        tool_orchestration_abuse = ToolOrchestrationAbuse(
            types=["cross_tool_state_leakage"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await tool_orchestration_abuse.a_assess(
            model_callback=dummy_model_callback,
        )

        assert tool_orchestration_abuse.is_vulnerable() is not None
        assert (
            tool_orchestration_abuse.simulated_attacks is not None
            and isinstance(tool_orchestration_abuse.simulated_attacks, dict)
        )
        assert tool_orchestration_abuse.res is not None and isinstance(
            tool_orchestration_abuse.res, dict
        )
        assert ToolOrchestrationAbuseType.CROSS_TOOL_STATE_LEAKAGE in results
        assert (
            len(results[ToolOrchestrationAbuseType.CROSS_TOOL_STATE_LEAKAGE])
            == 1
        )
        test_case = results[
            ToolOrchestrationAbuseType.CROSS_TOOL_STATE_LEAKAGE
        ][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
