import pytest
from deepteam.vulnerabilities import Hallucination
from deepteam.vulnerabilities.hallucination.types import HallucinationType
from deepteam.test_case import RTTestCase


class TestHallucination:
    def test_hallucination_all_types(self):
        types = [
            "fake_citations",
            "fake_apis",
            "fake_entities",
            "fake_statistics",
        ]
        hallucination = Hallucination(types=types)
        assert sorted(type.value for type in hallucination.types) == sorted(
            types
        )

    def test_hallucination_all_types_default(self):
        hallucination = Hallucination()
        assert sorted(type.value for type in hallucination.types) == sorted(
            type.value for type in HallucinationType
        )

    def test_hallucination_fake_citations(self):
        types = ["fake_citations"]
        hallucination = Hallucination(types=types)
        assert sorted(type.value for type in hallucination.types) == sorted(
            types
        )

    def test_hallucination_fake_apis(self):
        types = ["fake_apis"]
        hallucination = Hallucination(types=types)
        assert sorted(type.value for type in hallucination.types) == sorted(
            types
        )

    def test_hallucination_fake_entities(self):
        types = ["fake_entities"]
        hallucination = Hallucination(types=types)
        assert sorted(type.value for type in hallucination.types) == sorted(
            types
        )

    def test_hallucination_fake_statistics(self):
        types = ["fake_statistics"]
        hallucination = Hallucination(types=types)
        assert sorted(type.value for type in hallucination.types) == sorted(
            types
        )

    def test_hallucination_invalid_type(self):
        types = ["fake_citations", "invalid_type"]
        with pytest.raises(ValueError):
            Hallucination(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        hallucination = Hallucination(types=["fake_citations"])
        test_cases = hallucination.simulate_attacks(
            attacks_per_vulnerability_type=2
        )
        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Hallucination" for tc in test_cases)
        assert all(
            tc.vulnerability_type == HallucinationType.FAKE_CITATIONS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        hallucination = Hallucination(types=["fake_apis"], async_mode=False)

        def dummy_model_callback(prompt):
            return prompt

        results = hallucination.assess(
            model_callback=dummy_model_callback,
        )
        assert hallucination.is_vulnerable() is not None
        assert hallucination.simulated_attacks is not None and isinstance(
            hallucination.simulated_attacks, dict
        )
        assert hallucination.res is not None and isinstance(
            hallucination.res, dict
        )
        assert HallucinationType.FAKE_APIS in results
        assert len(results[HallucinationType.FAKE_APIS]) == 1
        test_case = results[HallucinationType.FAKE_APIS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_hallucination_metric(self):
        from deepteam.metrics import HallucinationMetric

        hallucination = Hallucination(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = hallucination._get_metric(HallucinationType.FAKE_CITATIONS)
        assert isinstance(metric, HallucinationMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        hallucination = Hallucination(types=["fake_entities"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await hallucination.a_assess(
            model_callback=dummy_model_callback,
        )
        assert hallucination.is_vulnerable() is not None
        assert hallucination.simulated_attacks is not None and isinstance(
            hallucination.simulated_attacks, dict
        )
        assert hallucination.res is not None and isinstance(
            hallucination.res, dict
        )
        assert HallucinationType.FAKE_ENTITIES in results
        assert len(results[HallucinationType.FAKE_ENTITIES]) == 1
        test_case = results[HallucinationType.FAKE_ENTITIES][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
