import pytest

from deepteam.vulnerabilities import BFLA
from deepteam.vulnerabilities.bfla import BFLAType
from deepteam.test_case import RTTestCase


class TestBFLA:

    def test_bfla_all_types(self):
        types = [
            "privilege_escalation",
            "function_bypass",
            "authorization_bypass",
        ]
        bfla = BFLA(types=types)
        assert sorted(type.value for type in bfla.types) == sorted(types)

    def test_bfla_all_types_default(self):
        bfla = BFLA()
        assert sorted(type.value for type in bfla.types) == sorted(
            type.value for type in BFLAType
        )

    def test_bfla_privilege_escalation(self):
        types = ["privilege_escalation"]
        bfla = BFLA(types=types)
        assert sorted(type.value for type in bfla.types) == sorted(types)

    def test_bfla_function_bypass(self):
        types = ["function_bypass"]
        bfla = BFLA(types=types)
        assert sorted(type.value for type in bfla.types) == sorted(types)

    def test_bfla_authorization_bypass(self):
        types = ["authorization_bypass"]
        bfla = BFLA(types=types)
        assert sorted(type.value for type in bfla.types) == sorted(types)

    def test_bfla_all_types_invalid(self):
        types = [
            "privilege_escalation",
            "function_bypass",
            "authorization_bypass",
            "invalid",
        ]
        with pytest.raises(ValueError):
            BFLA(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        bfla = BFLA(types=["authorization_bypass"])
        test_cases = bfla.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "BFLA" for tc in test_cases)
        assert all(
            tc.vulnerability_type == BFLAType.AUTHORIZATION_BYPASS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        bfla = BFLA(types=["authorization_bypass"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = bfla.assess(
            model_callback=dummy_model_callback,
        )

        assert bfla.is_vulnerable() is not None
        assert bfla.simulated_attacks is not None and isinstance(
            bfla.simulated_attacks, dict
        )
        assert bfla.res is not None and isinstance(bfla.res, dict)
        assert BFLAType.AUTHORIZATION_BYPASS in results
        assert len(results[BFLAType.AUTHORIZATION_BYPASS]) == 1
        test_case = results[BFLAType.AUTHORIZATION_BYPASS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_BFLA_metric(self):
        from deepteam.metrics import BFLAMetric

        bfla = BFLA(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = bfla._get_metric(BFLAType.AUTHORIZATION_BYPASS)
        assert isinstance(metric, BFLAMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        bfla = BFLA(types=["authorization_bypass"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await bfla.a_assess(
            model_callback=dummy_model_callback,
        )

        assert bfla.is_vulnerable() is not None
        assert bfla.simulated_attacks is not None and isinstance(
            bfla.simulated_attacks, dict
        )
        assert bfla.res is not None and isinstance(bfla.res, dict)
        assert BFLAType.AUTHORIZATION_BYPASS in results
        assert len(results[BFLAType.AUTHORIZATION_BYPASS]) == 1
        test_case = results[BFLAType.AUTHORIZATION_BYPASS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
