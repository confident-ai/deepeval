import pytest

from deepteam.vulnerabilities import IntellectualProperty
from deepteam.vulnerabilities.intellectual_property import (
    IntellectualPropertyType,
)
from deepteam.test_case import RTTestCase


class TestIntellectualProperty:

    def test_intellectual_property_all_types(self):
        types = [
            "imitation",
            "copyright_violations",
            "trademark_infringement",
            "patent_disclosure",
        ]
        intellectual_property = IntellectualProperty(types=types)
        assert sorted(
            type.value for type in intellectual_property.types
        ) == sorted(types)

    def test_intellectual_property_all_types_default(self):
        intellectual_property = IntellectualProperty()
        assert sorted(
            type.value for type in intellectual_property.types
        ) == sorted(type.value for type in IntellectualPropertyType)

    def test_intellectual_property_imitation(self):
        types = ["imitation"]
        intellectual_property = IntellectualProperty(types=types)
        assert sorted(
            type.value for type in intellectual_property.types
        ) == sorted(types)

    def test_intellectual_property_copyright_violations(self):
        types = ["copyright_violations"]
        intellectual_property = IntellectualProperty(types=types)
        assert sorted(
            type.value for type in intellectual_property.types
        ) == sorted(types)

    def test_intellectual_property_trademark_infringement(self):
        types = ["trademark_infringement"]
        intellectual_property = IntellectualProperty(types=types)
        assert sorted(
            type.value for type in intellectual_property.types
        ) == sorted(types)

    def test_intellectual_property_patent_disclosure(self):
        types = ["patent_disclosure"]
        intellectual_property = IntellectualProperty(types=types)
        assert sorted(
            type.value for type in intellectual_property.types
        ) == sorted(types)

    def test_intellectual_property_all_types_invalid(self):
        types = [
            "imitation",
            "copyright_violations",
            "trademark_infringement",
            "patent_disclosure",
            "invalid",
        ]
        with pytest.raises(ValueError):
            IntellectualProperty(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        intellectual = IntellectualProperty(types=["imitation"])
        test_cases = intellectual.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Intellectual Property" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type == IntellectualPropertyType.IMITATION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        intellectual = IntellectualProperty(
            types=["imitation"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = intellectual.assess(
            model_callback=dummy_model_callback,
        )

        assert intellectual.is_vulnerable() is not None
        assert intellectual.simulated_attacks is not None and isinstance(
            intellectual.simulated_attacks, dict
        )
        assert intellectual.res is not None and isinstance(
            intellectual.res, dict
        )
        assert IntellectualPropertyType.IMITATION in results
        assert len(results[IntellectualPropertyType.IMITATION]) == 1
        test_case = results[IntellectualPropertyType.IMITATION][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_IntellectualProperty_metric(self):
        from deepteam.metrics import IntellectualPropertyMetric

        intellectual = IntellectualProperty(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = intellectual._get_metric(IntellectualPropertyType.IMITATION)
        assert isinstance(metric, IntellectualPropertyMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        intellectual = IntellectualProperty(
            types=["imitation"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await intellectual.a_assess(
            model_callback=dummy_model_callback,
        )

        assert intellectual.is_vulnerable() is not None
        assert intellectual.simulated_attacks is not None and isinstance(
            intellectual.simulated_attacks, dict
        )
        assert intellectual.res is not None and isinstance(
            intellectual.res, dict
        )
        assert IntellectualPropertyType.IMITATION in results
        assert len(results[IntellectualPropertyType.IMITATION]) == 1
        test_case = results[IntellectualPropertyType.IMITATION][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
