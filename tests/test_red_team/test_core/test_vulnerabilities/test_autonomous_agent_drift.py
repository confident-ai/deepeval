import pytest

from deepteam.vulnerabilities import AutonomousAgentDrift
from deepteam.vulnerabilities.autonomous_agent_drift import (
    AutonomousAgentDriftType,
)
from deepteam.test_case import RTTestCase


class TestAutonomousAgentDrift:

    def test_autonomous_agent_drift_all_types(self):
        types = [
            "goal_drift",
            "reward_hacking",
            "agent_collusion",
            "runaway_autonomy",
        ]
        autonoumus_agent_drift = AutonomousAgentDrift(types=types)
        assert sorted(
            type.value for type in autonoumus_agent_drift.types
        ) == sorted(types)

    def test_autonomous_agent_drift_all_types_default(self):
        autonoumus_agent_drift = AutonomousAgentDrift()
        assert sorted(
            type.value for type in autonoumus_agent_drift.types
        ) == sorted(type.value for type in AutonomousAgentDriftType)

    def test_autonomous_agent_drift_goal_drift(self):
        types = ["goal_drift"]
        autonoumus_agent_drift = AutonomousAgentDrift(types=types)
        assert sorted(
            type.value for type in autonoumus_agent_drift.types
        ) == sorted(types)

    def test_autonomous_agent_drift_reward_hacking(self):
        types = ["reward_hacking"]
        autonoumus_agent_drift = AutonomousAgentDrift(types=types)
        assert sorted(
            type.value for type in autonoumus_agent_drift.types
        ) == sorted(types)

    def test_autonomous_agent_drift_agent_collusion(self):
        types = ["agent_collusion"]
        autonoumus_agent_drift = AutonomousAgentDrift(types=types)
        assert sorted(
            type.value for type in autonoumus_agent_drift.types
        ) == sorted(types)

    def test_autonomous_agent_drift_runaway_autonomy(self):
        types = ["runaway_autonomy"]
        autonoumus_agent_drift = AutonomousAgentDrift(types=types)
        assert sorted(
            type.value for type in autonoumus_agent_drift.types
        ) == sorted(types)

    def test_autonomous_agent_drift_all_types_invalid(self):
        types = [
            "goal_drift",
            "reward_hacking",
            "agent_collusion",
            "runaway_autonomy",
            "invalid",
        ]
        with pytest.raises(ValueError):
            AutonomousAgentDrift(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        autonoumus_agent_drift = AutonomousAgentDrift(types=["goal_drift"])
        test_cases = autonoumus_agent_drift.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Autonomous Agent Drift" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type == AutonomousAgentDriftType.GOAL_DRIFT
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        autonoumus_agent_drift = AutonomousAgentDrift(
            types=["reward_hacking"], async_mode=False
        )

        def dummy_model_callback(prompt):
            return prompt

        results = autonoumus_agent_drift.assess(
            model_callback=dummy_model_callback,
        )

        assert autonoumus_agent_drift.is_vulnerable() is not None
        assert (
            autonoumus_agent_drift.simulated_attacks is not None
            and isinstance(autonoumus_agent_drift.simulated_attacks, dict)
        )
        assert autonoumus_agent_drift.res is not None and isinstance(
            autonoumus_agent_drift.res, dict
        )
        assert AutonomousAgentDriftType.REWARD_HACKING in results
        assert len(results[AutonomousAgentDriftType.REWARD_HACKING]) == 1
        test_case = results[AutonomousAgentDriftType.REWARD_HACKING][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_tool_metadata_poisoning_metric(self):
        from deepteam.metrics import AutonomousAgentDriftMetric

        autonoumus_agent_drift = AutonomousAgentDrift(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = autonoumus_agent_drift._get_metric(
            AutonomousAgentDriftType.REWARD_HACKING
        )
        assert isinstance(metric, AutonomousAgentDriftMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        autonoumus_agent_drift = AutonomousAgentDrift(
            types=["runaway_autonomy"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await autonoumus_agent_drift.a_assess(
            model_callback=dummy_model_callback,
        )

        assert autonoumus_agent_drift.is_vulnerable() is not None
        assert (
            autonoumus_agent_drift.simulated_attacks is not None
            and isinstance(autonoumus_agent_drift.simulated_attacks, dict)
        )
        assert autonoumus_agent_drift.res is not None and isinstance(
            autonoumus_agent_drift.res, dict
        )
        assert AutonomousAgentDriftType.RUNAWAY_AUTONOMY in results
        assert len(results[AutonomousAgentDriftType.RUNAWAY_AUTONOMY]) == 1
        test_case = results[AutonomousAgentDriftType.RUNAWAY_AUTONOMY][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
