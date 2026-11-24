import pytest

from deepteam.vulnerabilities import Bias
from deepteam.vulnerabilities.bias import BiasType
from deepteam.test_case import RTTestCase


class TestBias:

    def test_bias_all_types(self):
        types = ["gender", "politics", "religion", "race"]
        bias = Bias(types=types)
        assert sorted(type.value for type in bias.types) == sorted(types)

    def test_bias_all_types_default(self):
        bias = Bias()
        assert sorted(type.value for type in bias.types) == sorted(
            type.value for type in BiasType
        )

    def test_bias_gender(self):
        types = ["gender"]
        bias = Bias(types=types)
        assert sorted(type.value for type in bias.types) == sorted(types)

    def test_bias_politics(self):
        types = ["politics"]
        bias = Bias(types=types)
        assert sorted(type.value for type in bias.types) == sorted(types)

    def test_bias_religion(self):
        types = ["religion"]
        bias = Bias(types=types)
        assert sorted(type.value for type in bias.types) == sorted(types)

    def test_bias_race(self):
        types = ["race"]
        bias = Bias(types=types)
        assert sorted(type.value for type in bias.types) == sorted(types)

    def test_bias_all_types_invalid(self):
        types = ["gender", "politics", "religion", "race", "invalid"]
        with pytest.raises(ValueError):
            Bias(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        bias = Bias(types=["gender"])
        test_cases = bias.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Bias" for tc in test_cases)
        assert all(
            tc.vulnerability_type == BiasType.GENDER for tc in test_cases
        )

    def test_assess_returns_results(self):
        bias = Bias(types=["politics"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = bias.assess(
            model_callback=dummy_model_callback,
        )

        assert bias.is_vulnerable() is not None
        assert bias.simulated_attacks is not None and isinstance(
            bias.simulated_attacks, dict
        )
        assert bias.res is not None and isinstance(bias.res, dict)
        assert BiasType.POLITICS in results
        assert len(results[BiasType.POLITICS]) == 1
        test_case = results[BiasType.POLITICS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_bias_metric(self):
        from deepteam.metrics import BiasMetric

        bias = Bias(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = bias._get_metric(BiasType.RACE)
        assert isinstance(metric, BiasMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        bias = Bias(types=["religion"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await bias.a_assess(
            model_callback=dummy_model_callback,
        )

        assert bias.is_vulnerable() is not None
        assert bias.simulated_attacks is not None and isinstance(
            bias.simulated_attacks, dict
        )
        assert bias.res is not None and isinstance(bias.res, dict)
        assert BiasType.RELIGION in results
        assert len(results[BiasType.RELIGION]) == 1
        test_case = results[BiasType.RELIGION][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
