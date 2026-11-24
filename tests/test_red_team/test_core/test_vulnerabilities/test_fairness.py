import pytest

from deepteam.vulnerabilities import Fairness
from deepteam.vulnerabilities.fairness import FairnessType
from deepteam.test_case import RTTestCase


class TestFairness:

    def test_fairness_all_types(self):
        types = [
            "equality_consistency",
            "procedural_opportunity",
            "temporal_outcome",
        ]
        fairness = Fairness(types=types)
        assert sorted(type.value for type in fairness.types) == sorted(types)

    def test_fairness_all_types_default(self):
        fairness = Fairness()
        assert sorted(type.value for type in fairness.types) == sorted(
            type.value for type in FairnessType
        )

    def test_fairness_equality_consistency(self):
        types = ["equality_consistency"]
        fairness = Fairness(types=types)
        assert sorted(type.value for type in fairness.types) == sorted(types)

    def test_fairness_procedural_opportunity(self):
        types = ["procedural_opportunity"]
        fairness = Fairness(types=types)
        assert sorted(type.value for type in fairness.types) == sorted(types)

    def test_fairness_temporal_outcome(self):
        types = ["temporal_outcome"]
        fairness = Fairness(types=types)
        assert sorted(type.value for type in fairness.types) == sorted(types)

    def test_fairness_all_types_invalid(self):
        types = [
            "equality_consistency",
            "procedural_opportunity",
            "temporal_outcome",
            "invalid",
        ]
        with pytest.raises(ValueError):
            Fairness(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        fairness = Fairness(types=["equality_consistency"])
        test_cases = fairness.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Fairness" for tc in test_cases)
        assert all(
            tc.vulnerability_type == FairnessType.EQUALITY_CONSISTENCY
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        fairness = Fairness(types=["procedural_opportunity"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = fairness.assess(
            model_callback=dummy_model_callback,
        )

        assert fairness.is_vulnerable() is not None
        assert fairness.simulated_attacks is not None and isinstance(
            fairness.simulated_attacks, dict
        )
        assert fairness.res is not None and isinstance(fairness.res, dict)
        assert FairnessType.PROCEDURAL_OPPORTUNITY in results
        assert len(results[FairnessType.PROCEDURAL_OPPORTUNITY]) == 1
        test_case = results[FairnessType.PROCEDURAL_OPPORTUNITY][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_fairness_metric(self):
        from deepteam.metrics import FairnessMetric

        fairness = Fairness(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = fairness._get_metric(FairnessType.PROCEDURAL_OPPORTUNITY)
        assert isinstance(metric, FairnessMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        fairness = Fairness(types=["equality_consistency"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await fairness.a_assess(
            model_callback=dummy_model_callback,
        )

        assert fairness.is_vulnerable() is not None
        assert fairness.simulated_attacks is not None and isinstance(
            fairness.simulated_attacks, dict
        )
        assert fairness.res is not None and isinstance(fairness.res, dict)
        assert FairnessType.EQUALITY_CONSISTENCY in results
        assert len(results[FairnessType.EQUALITY_CONSISTENCY]) == 1
        test_case = results[FairnessType.EQUALITY_CONSISTENCY][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
