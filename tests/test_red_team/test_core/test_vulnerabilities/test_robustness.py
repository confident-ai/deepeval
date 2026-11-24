import pytest

from deepteam.vulnerabilities import Robustness
from deepteam.vulnerabilities.robustness import RobustnessType
from deepteam.test_case import RTTestCase


class TestRobustness:

    def test_robustness_all_types(self):
        types = ["input_overreliance", "hijacking"]
        robustness = Robustness(types=types)
        assert sorted(type.value for type in robustness.types) == sorted(types)

    def test_robustness_all_types_default(self):
        robustness = Robustness()
        assert sorted(type.value for type in robustness.types) == sorted(
            type.value for type in RobustnessType
        )

    def test_robustness_input_overreliance(self):
        types = ["input_overreliance"]
        robustness = Robustness(types=types)
        assert sorted(type.value for type in robustness.types) == sorted(types)

    def test_robustness_hijacking(self):
        types = ["hijacking"]
        robustness = Robustness(types=types)
        assert sorted(type.value for type in robustness.types) == sorted(types)

    def test_robustness_all_types_invalid(self):
        types = ["input_overreliance", "hijacking", "invalid"]
        with pytest.raises(ValueError):
            Robustness(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        robustness = Robustness(types=["hijacking"])
        test_cases = robustness.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Robustness" for tc in test_cases)
        assert all(
            tc.vulnerability_type == RobustnessType.HIJACKING
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        robustness = Robustness(types=["hijacking"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = robustness.assess(
            model_callback=dummy_model_callback,
        )

        assert robustness.is_vulnerable() is not None
        assert robustness.simulated_attacks is not None and isinstance(
            robustness.simulated_attacks, dict
        )
        assert robustness.res is not None and isinstance(robustness.res, dict)
        assert RobustnessType.HIJACKING in results
        assert len(results[RobustnessType.HIJACKING]) == 1
        test_case = results[RobustnessType.HIJACKING][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_Robustness_overreliance_metric(self):
        from deepteam.metrics import OverrelianceMetric

        robustness = Robustness(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = robustness._get_metric(RobustnessType.INPUT_OVERRELIANCE)
        assert isinstance(metric, OverrelianceMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    def test_get_metric_returns_Robustness_hijacking_metric(self):
        from deepteam.metrics import HijackingMetric

        robustness = Robustness(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = robustness._get_metric(RobustnessType.HIJACKING)
        assert isinstance(metric, HijackingMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        robustness = Robustness(types=["hijacking"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await robustness.a_assess(
            model_callback=dummy_model_callback,
        )

        assert robustness.is_vulnerable() is not None
        assert robustness.simulated_attacks is not None and isinstance(
            robustness.simulated_attacks, dict
        )
        assert robustness.res is not None and isinstance(robustness.res, dict)
        assert RobustnessType.HIJACKING in results
        assert len(results[RobustnessType.HIJACKING]) == 1
        test_case = results[RobustnessType.HIJACKING][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
