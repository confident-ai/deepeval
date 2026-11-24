import pytest

from deepteam.vulnerabilities import ExcessiveAgency
from deepteam.vulnerabilities.excessive_agency import (
    ExcessiveAgencyType,
)
from deepteam.test_case import RTTestCase


class TestExcessiveAgency:

    def test_excessive_agency_all_types(self):
        types = ["functionality", "permissions", "autonomy"]
        excessive_agency = ExcessiveAgency(types=types)
        assert sorted(type.value for type in excessive_agency.types) == sorted(
            types
        )

    def test_excessive_agency_all_types_default(self):
        excessive_agency = ExcessiveAgency()
        assert sorted(type.value for type in excessive_agency.types) == sorted(
            type.value for type in ExcessiveAgencyType
        )

    def test_excessive_agency_functionality(self):
        types = ["functionality"]
        excessive_agency = ExcessiveAgency(types=types)
        assert sorted(type.value for type in excessive_agency.types) == sorted(
            types
        )

    def test_excessive_agency_permissions(self):
        types = ["permissions"]
        excessive_agency = ExcessiveAgency(types=types)
        assert sorted(type.value for type in excessive_agency.types) == sorted(
            types
        )

    def test_excessive_agency_autonomy(self):
        types = ["autonomy"]
        excessive_agency = ExcessiveAgency(types=types)
        assert sorted(type.value for type in excessive_agency.types) == sorted(
            types
        )

    def test_excessive_agency_all_types_invalid(self):
        types = ["functionality", "permissions", "autonomy", "invalid"]
        with pytest.raises(ValueError):
            ExcessiveAgency(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        excessive_agency = ExcessiveAgency(types=["autonomy"])
        test_cases = excessive_agency.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Excessive Agency" for tc in test_cases)
        assert all(
            tc.vulnerability_type == ExcessiveAgencyType.AUTONOMY
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        excessive_agency = ExcessiveAgency(types=["autonomy"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = excessive_agency.assess(
            model_callback=dummy_model_callback,
        )

        assert excessive_agency.is_vulnerable() is not None
        assert excessive_agency.simulated_attacks is not None and isinstance(
            excessive_agency.simulated_attacks, dict
        )
        assert excessive_agency.res is not None and isinstance(
            excessive_agency.res, dict
        )
        assert ExcessiveAgencyType.AUTONOMY in results
        assert len(results[ExcessiveAgencyType.AUTONOMY]) == 1
        test_case = results[ExcessiveAgencyType.AUTONOMY][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_ExcessiveAgency_metric(self):
        from deepteam.metrics import ExcessiveAgencyMetric

        excessive_agency = ExcessiveAgency(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = excessive_agency._get_metric(ExcessiveAgencyType.AUTONOMY)
        assert isinstance(metric, ExcessiveAgencyMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        excessive_agency = ExcessiveAgency(types=["autonomy"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await excessive_agency.a_assess(
            model_callback=dummy_model_callback,
        )

        assert excessive_agency.is_vulnerable() is not None
        assert excessive_agency.simulated_attacks is not None and isinstance(
            excessive_agency.simulated_attacks, dict
        )
        assert excessive_agency.res is not None and isinstance(
            excessive_agency.res, dict
        )
        assert ExcessiveAgencyType.AUTONOMY in results
        assert len(results[ExcessiveAgencyType.AUTONOMY]) == 1
        test_case = results[ExcessiveAgencyType.AUTONOMY][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
