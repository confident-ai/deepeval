import pytest

from deepteam.vulnerabilities import Toxicity
from deepteam.vulnerabilities.toxicity import ToxicityType
from deepteam.test_case import RTTestCase


class TestToxicity:

    def test_toxicity_all_types(self):
        types = ["profanity", "insults", "threats", "mockery"]
        toxicity = Toxicity(types=types)
        assert sorted(type.value for type in toxicity.types) == sorted(types)

    def test_toxicity_all_types_default(self):
        toxicity = Toxicity()
        assert sorted(type.value for type in toxicity.types) == sorted(
            type.value for type in ToxicityType
        )

    def test_toxicity_profanity(self):
        types = ["profanity"]
        toxicity = Toxicity(types=types)
        assert sorted(type.value for type in toxicity.types) == sorted(types)

    def test_toxicity_insults(self):
        types = ["insults"]
        toxicity = Toxicity(types=types)
        assert sorted(type.value for type in toxicity.types) == sorted(types)

    def test_toxicity_threats(self):
        types = ["threats"]
        toxicity = Toxicity(types=types)
        assert sorted(type.value for type in toxicity.types) == sorted(types)

    def test_toxicity_mockery(self):
        types = ["mockery"]
        toxicity = Toxicity(types=types)
        assert sorted(type.value for type in toxicity.types) == sorted(types)

    def test_toxicity_all_types_invalid(self):
        types = ["profanity", "insults", "threats", "mockery", "invalid"]
        with pytest.raises(ValueError):
            Toxicity(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        toxicity = Toxicity(types=["insults"])
        test_cases = toxicity.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Toxicity" for tc in test_cases)
        assert all(
            tc.vulnerability_type == ToxicityType.INSULTS for tc in test_cases
        )

    def test_assess_returns_results(self):
        toxicity = Toxicity(types=["insults"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = toxicity.assess(
            model_callback=dummy_model_callback,
        )

        assert toxicity.is_vulnerable() is not None
        assert toxicity.simulated_attacks is not None and isinstance(
            toxicity.simulated_attacks, dict
        )
        assert toxicity.res is not None and isinstance(toxicity.res, dict)
        assert ToxicityType.INSULTS in results
        assert len(results[ToxicityType.INSULTS]) == 1
        test_case = results[ToxicityType.INSULTS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_Toxicity_metric(self):
        from deepteam.metrics import ToxicityMetric

        toxicity = Toxicity(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = toxicity._get_metric(ToxicityType.INSULTS)
        assert isinstance(metric, ToxicityMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        toxicity = Toxicity(types=["insults"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await toxicity.a_assess(
            model_callback=dummy_model_callback,
        )

        assert toxicity.is_vulnerable() is not None
        assert toxicity.simulated_attacks is not None and isinstance(
            toxicity.simulated_attacks, dict
        )
        assert toxicity.res is not None and isinstance(toxicity.res, dict)
        assert ToxicityType.INSULTS in results
        assert len(results[ToxicityType.INSULTS]) == 1
        test_case = results[ToxicityType.INSULTS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
