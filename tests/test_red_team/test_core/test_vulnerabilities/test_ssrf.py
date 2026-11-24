import pytest

from deepteam.vulnerabilities import SSRF
from deepteam.vulnerabilities.ssrf import SSRFType
from deepteam.test_case import RTTestCase


class TestSSRF:

    def test_ssrf_all_types(self):
        types = [
            "internal_service_access",
            "cloud_metadata_access",
            "port_scanning",
        ]
        ssrf = SSRF(types=types)
        assert sorted(type.value for type in ssrf.types) == sorted(types)

    def test_ssrf_all_types_default(self):
        ssrf = SSRF()
        assert sorted(type.value for type in ssrf.types) == sorted(
            type.value for type in SSRFType
        )

    def test_ssrf_internal_service_access(self):
        types = ["internal_service_access"]
        ssrf = SSRF(types=types)
        assert sorted(type.value for type in ssrf.types) == sorted(types)

    def test_ssrf_cloud_metadata_access(self):
        types = ["cloud_metadata_access"]
        ssrf = SSRF(types=types)
        assert sorted(type.value for type in ssrf.types) == sorted(types)

    def test_ssrf_port_scanning(self):
        types = ["port_scanning"]
        ssrf = SSRF(types=types)
        assert sorted(type.value for type in ssrf.types) == sorted(types)

    def test_ssrf_all_types_invalid(self):
        types = [
            "internal_service_access",
            "cloud_metadata_access",
            "port_scanning",
            "invalid",
        ]
        with pytest.raises(ValueError):
            SSRF(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        ssrf = SSRF(types=["cloud_metadata_access"])
        test_cases = ssrf.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "SSRF" for tc in test_cases)
        assert all(
            tc.vulnerability_type == SSRFType.CLOUD_METADATA_ACCESS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        ssrf = SSRF(types=["cloud_metadata_access"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = ssrf.assess(
            model_callback=dummy_model_callback,
        )

        assert ssrf.is_vulnerable() is not None
        assert ssrf.simulated_attacks is not None and isinstance(
            ssrf.simulated_attacks, dict
        )
        assert ssrf.res is not None and isinstance(ssrf.res, dict)
        assert SSRFType.CLOUD_METADATA_ACCESS in results
        assert len(results[SSRFType.CLOUD_METADATA_ACCESS]) == 1
        test_case = results[SSRFType.CLOUD_METADATA_ACCESS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_SSRF_metric(self):
        from deepteam.metrics import SSRFMetric

        ssrf = SSRF(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = ssrf._get_metric(SSRFType.CLOUD_METADATA_ACCESS)
        assert isinstance(metric, SSRFMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        ssrf = SSRF(types=["cloud_metadata_access"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await ssrf.a_assess(
            model_callback=dummy_model_callback,
        )

        assert ssrf.is_vulnerable() is not None
        assert ssrf.simulated_attacks is not None and isinstance(
            ssrf.simulated_attacks, dict
        )
        assert ssrf.res is not None and isinstance(ssrf.res, dict)
        assert SSRFType.CLOUD_METADATA_ACCESS in results
        assert len(results[SSRFType.CLOUD_METADATA_ACCESS]) == 1
        test_case = results[SSRFType.CLOUD_METADATA_ACCESS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
