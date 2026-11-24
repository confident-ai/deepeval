import pytest

from deepteam.vulnerabilities import Ethics
from deepteam.vulnerabilities.ethics import EthicsType
from deepteam.test_case import RTTestCase


class TestEthics:

    def test_ethics_all_types(self):
        types = [
            "harm_prevention",
            "moral_integrity",
            "responsible_transparency",
        ]
        ethics = Ethics(types=types)
        assert sorted(type.value for type in ethics.types) == sorted(types)

    def test_ethics_all_types_default(self):
        ethics = Ethics()
        assert sorted(type.value for type in ethics.types) == sorted(
            type.value for type in EthicsType
        )

    def test_ethics_harm_prevention(self):
        types = ["harm_prevention"]
        ethics = Ethics(types=types)
        assert sorted(type.value for type in ethics.types) == sorted(types)

    def test_ethics_moral_integrity(self):
        types = ["moral_integrity"]
        ethics = Ethics(types=types)
        assert sorted(type.value for type in ethics.types) == sorted(types)

    def test_ethics_responsible_transparency(self):
        types = ["responsible_transparency"]
        ethics = Ethics(types=types)
        assert sorted(type.value for type in ethics.types) == sorted(types)

    def test_ethics_all_types_invalid(self):
        types = [
            "harm_prevention",
            "moral_integrity",
            "responsible_transparency",
            "invalid",
        ]
        with pytest.raises(ValueError):
            Ethics(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        ethics = Ethics(types=["harm_prevention"])
        test_cases = ethics.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Ethics" for tc in test_cases)
        assert all(
            tc.vulnerability_type == EthicsType.HARM_PREVENTION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        ethics = Ethics(types=["moral_integrity"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = ethics.assess(
            model_callback=dummy_model_callback,
        )

        assert ethics.is_vulnerable() is not None
        assert ethics.simulated_attacks is not None and isinstance(
            ethics.simulated_attacks, dict
        )
        assert ethics.res is not None and isinstance(ethics.res, dict)
        assert EthicsType.MORAL_INTEGRITY in results
        assert len(results[EthicsType.MORAL_INTEGRITY]) == 1
        test_case = results[EthicsType.MORAL_INTEGRITY][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_ethics_metric(self):
        from deepteam.metrics import EthicsMetric

        ethics = Ethics(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = ethics._get_metric(EthicsType.HARM_PREVENTION)
        assert isinstance(metric, EthicsMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        ethics = Ethics(types=["moral_integrity"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await ethics.a_assess(
            model_callback=dummy_model_callback,
        )

        assert ethics.is_vulnerable() is not None
        assert ethics.simulated_attacks is not None and isinstance(
            ethics.simulated_attacks, dict
        )
        assert ethics.res is not None and isinstance(ethics.res, dict)
        assert EthicsType.MORAL_INTEGRITY in results
        assert len(results[EthicsType.MORAL_INTEGRITY]) == 1
        test_case = results[EthicsType.MORAL_INTEGRITY][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
