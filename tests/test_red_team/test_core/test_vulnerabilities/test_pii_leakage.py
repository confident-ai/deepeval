import pytest

from deepteam.vulnerabilities import PIILeakage
from deepteam.vulnerabilities.pii_leakage import PIILeakageType
from deepteam.test_case import RTTestCase


class TestPIILeakage:

    def test_pii_leakage_all_types(self):
        types = [
            "api_and_database_access",
            "direct_disclosure",
            "session_leak",
            "social_manipulation",
        ]
        pii_leakage = PIILeakage(types=types)
        assert sorted(type.value for type in pii_leakage.types) == sorted(types)

    def test_pii_leakage_all_types_default(self):
        pii_leakage = PIILeakage()
        assert sorted(type.value for type in pii_leakage.types) == sorted(
            type.value for type in PIILeakageType
        )

    def test_pii_leakage_database_access(self):
        types = ["api_and_database_access"]
        pii_leakage = PIILeakage(types=types)
        assert sorted(type.value for type in pii_leakage.types) == sorted(types)

    def test_pii_leakage_direct_disclosure(self):
        types = ["direct_disclosure"]
        pii_leakage = PIILeakage(types=types)
        assert sorted(type.value for type in pii_leakage.types) == sorted(types)

    def test_pii_leakage_session_leak(self):
        types = ["session_leak"]
        pii_leakage = PIILeakage(types=types)
        assert sorted(type.value for type in pii_leakage.types) == sorted(types)

    def test_pii_leakage_social_manipulation(self):
        types = ["social_manipulation"]
        pii_leakage = PIILeakage(types=types)
        assert sorted(type.value for type in pii_leakage.types) == sorted(types)

    def test_pii_leakage_all_types_invalid(self):
        types = [
            "api_and_database_access",
            "direct_disclosure",
            "session_leak",
            "social_manipulation",
            "invalid",
        ]
        with pytest.raises(ValueError):
            PIILeakage(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        pii_lekage = PIILeakage(types=["session_leak"])
        test_cases = pii_lekage.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "PII Leakage" for tc in test_cases)
        assert all(
            tc.vulnerability_type == PIILeakageType.SESSION_LEAK
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        pii_lekage = PIILeakage(types=["session_leak"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = pii_lekage.assess(
            model_callback=dummy_model_callback,
        )

        assert pii_lekage.is_vulnerable() is not None
        assert pii_lekage.simulated_attacks is not None and isinstance(
            pii_lekage.simulated_attacks, dict
        )
        assert pii_lekage.res is not None and isinstance(pii_lekage.res, dict)
        assert PIILeakageType.SESSION_LEAK in results
        assert len(results[PIILeakageType.SESSION_LEAK]) == 1
        test_case = results[PIILeakageType.SESSION_LEAK][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_PIILeakage_metric(self):
        from deepteam.metrics import PIIMetric

        pii_lekage = PIILeakage(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = pii_lekage._get_metric(PIILeakageType.SESSION_LEAK)
        assert isinstance(metric, PIIMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        pii_lekage = PIILeakage(types=["session_leak"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await pii_lekage.a_assess(
            model_callback=dummy_model_callback,
        )

        assert pii_lekage.is_vulnerable() is not None
        assert pii_lekage.simulated_attacks is not None and isinstance(
            pii_lekage.simulated_attacks, dict
        )
        assert pii_lekage.res is not None and isinstance(pii_lekage.res, dict)
        assert PIILeakageType.SESSION_LEAK in results
        assert len(results[PIILeakageType.SESSION_LEAK]) == 1
        test_case = results[PIILeakageType.SESSION_LEAK][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
