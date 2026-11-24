import pytest

from deepteam.vulnerabilities import Misinformation
from deepteam.vulnerabilities.misinformation import MisinformationType
from deepteam.test_case import RTTestCase


class TestMisinformation:

    def test_misinformation_all_types(self):
        types = [
            "factual_errors",
            "unsupported_claims",
            "expertize_misrepresentation",
        ]
        misinformation = Misinformation(
            types=types,
        )
        assert sorted(type.value for type in misinformation.types) == sorted(
            types
        )

    def test_misinformation_all_types_default(self):
        misinformation = Misinformation()
        assert sorted(type.value for type in misinformation.types) == sorted(
            type.value for type in MisinformationType
        )

    def test_misinformation_factual_errors(self):
        types = ["factual_errors"]
        misinformation = Misinformation(
            types=types,
        )
        assert sorted(type.value for type in misinformation.types) == sorted(
            types
        )

    def test_misinformation_unsupported_claims(self):
        types = ["unsupported_claims"]
        misinformation = Misinformation(
            types=types,
        )
        assert sorted(type.value for type in misinformation.types) == sorted(
            types
        )

    def test_misinformation_expertize_misrepresentation(self):
        types = ["expertize_misrepresentation"]
        misinformation = Misinformation(
            types=types,
        )
        assert sorted(type.value for type in misinformation.types) == sorted(
            types
        )

    def test_misinformation_all_types_invalid(self):
        types = [
            "factual_errors",
            "unsupported_claims",
            "expertize_misrepresentation",
            "invalid",
        ]
        with pytest.raises(ValueError):
            Misinformation(
                types=types,
            )

    def test_simulate_attacks_returns_expected_cases(self):
        misinformation = Misinformation(types=["factual_errors"])
        test_cases = misinformation.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Misinformation" for tc in test_cases)
        assert all(
            tc.vulnerability_type == MisinformationType.FACTUAL_ERRORS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        misinformation = Misinformation(
            types=["factual_errors"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = misinformation.assess(
            model_callback=dummy_model_callback,
        )

        assert misinformation.is_vulnerable() is not None
        assert misinformation.simulated_attacks is not None and isinstance(
            misinformation.simulated_attacks, dict
        )
        assert misinformation.res is not None and isinstance(
            misinformation.res, dict
        )
        assert MisinformationType.FACTUAL_ERRORS in results
        assert len(results[MisinformationType.FACTUAL_ERRORS]) == 1
        test_case = results[MisinformationType.FACTUAL_ERRORS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_Misinformation_metric(self):
        from deepteam.metrics import MisinformationMetric

        misinformation = Misinformation(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = misinformation._get_metric(MisinformationType.FACTUAL_ERRORS)
        assert isinstance(metric, MisinformationMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        misinformation = Misinformation(
            types=["factual_errors"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await misinformation.a_assess(
            model_callback=dummy_model_callback,
        )

        assert misinformation.is_vulnerable() is not None
        assert misinformation.simulated_attacks is not None and isinstance(
            misinformation.simulated_attacks, dict
        )
        assert misinformation.res is not None and isinstance(
            misinformation.res, dict
        )
        assert MisinformationType.FACTUAL_ERRORS in results
        assert len(results[MisinformationType.FACTUAL_ERRORS]) == 1
        test_case = results[MisinformationType.FACTUAL_ERRORS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
