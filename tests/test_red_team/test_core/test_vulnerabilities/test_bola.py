import pytest

from deepteam.vulnerabilities import BOLA
from deepteam.vulnerabilities.bola import BOLAType
from deepteam.test_case import RTTestCase


class TestBOLA:

    def test_bola_all_types(self):
        types = [
            "object_access_bypass",
            "cross_customer_access",
            "unauthorized_object_manipulation",
        ]
        bola = BOLA(types=types)
        assert sorted(type.value for type in bola.types) == sorted(types)

    def test_bola_all_types_default(self):
        bola = BOLA()
        assert sorted(type.value for type in bola.types) == sorted(
            type.value for type in BOLAType
        )

    def test_bola_object_access_bypass(self):
        types = ["object_access_bypass"]
        bola = BOLA(types=types)
        assert sorted(type.value for type in bola.types) == sorted(types)

    def test_bola_cross_customer_access(self):
        types = ["cross_customer_access"]
        bola = BOLA(types=types)
        assert sorted(type.value for type in bola.types) == sorted(types)

    def test_bola_unauthorized_object_manipulation(self):
        types = ["unauthorized_object_manipulation"]
        bola = BOLA(types=types)
        assert sorted(type.value for type in bola.types) == sorted(types)

    def test_bola_all_types_invalid(self):
        types = [
            "object_access_bypass",
            "cross_customer_access",
            "unauthorized_object_manipulation",
            "invalid",
        ]
        with pytest.raises(ValueError):
            BOLA(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        bola = BOLA(types=["cross_customer_access"])
        test_cases = bola.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "BOLA" for tc in test_cases)
        assert all(
            tc.vulnerability_type == BOLAType.CROSS_CUSTOMER_ACCESS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        bola = BOLA(types=["cross_customer_access"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = bola.assess(
            model_callback=dummy_model_callback,
        )

        assert bola.is_vulnerable() is not None
        assert bola.simulated_attacks is not None and isinstance(
            bola.simulated_attacks, dict
        )
        assert bola.res is not None and isinstance(bola.res, dict)
        assert BOLAType.CROSS_CUSTOMER_ACCESS in results
        assert len(results[BOLAType.CROSS_CUSTOMER_ACCESS]) == 1
        test_case = results[BOLAType.CROSS_CUSTOMER_ACCESS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_BOLA_metric(self):
        from deepteam.metrics import BOLAMetric

        bola = BOLA(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = bola._get_metric(BOLAType.CROSS_CUSTOMER_ACCESS)
        assert isinstance(metric, BOLAMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        bola = BOLA(types=["cross_customer_access"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await bola.a_assess(
            model_callback=dummy_model_callback,
        )

        assert bola.is_vulnerable() is not None
        assert bola.simulated_attacks is not None and isinstance(
            bola.simulated_attacks, dict
        )
        assert bola.res is not None and isinstance(bola.res, dict)
        assert BOLAType.CROSS_CUSTOMER_ACCESS in results
        assert len(results[BOLAType.CROSS_CUSTOMER_ACCESS]) == 1
        test_case = results[BOLAType.CROSS_CUSTOMER_ACCESS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
