import pytest

from deepteam.vulnerabilities import ChildProtection
from deepteam.vulnerabilities.child_protection import ChildProtectionType
from deepteam.test_case import RTTestCase


class TestChildProtection:

    def test_child_protection_all_types(self):
        types = ["age_verification", "data_privacy", "exposure_interaction"]
        child_protection = ChildProtection(types=types)
        assert sorted(type.value for type in child_protection.types) == sorted(
            types
        )

    def test_child_protection_all_types_default(self):
        child_protection = ChildProtection()
        assert sorted(type.value for type in child_protection.types) == sorted(
            type.value for type in ChildProtectionType
        )

    def test_child_protection_age_verification(self):
        types = ["age_verification"]
        child_protection = ChildProtection(types=types)
        assert sorted(type.value for type in child_protection.types) == sorted(
            types
        )

    def test_child_protection_data_privacy(self):
        types = ["data_privacy"]
        child_protection = ChildProtection(types=types)
        assert sorted(type.value for type in child_protection.types) == sorted(
            types
        )

    def test_child_protection_exposure_interaction(self):
        types = ["exposure_interaction"]
        child_protection = ChildProtection(types=types)
        assert sorted(type.value for type in child_protection.types) == sorted(
            types
        )

    def test_child_protection_all_types_invalid(self):
        types = [
            "age_verification",
            "data_privacy",
            "exposure_interaction",
            "invalid",
        ]
        with pytest.raises(ValueError):
            ChildProtection(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        child_protection = ChildProtection(types=["age_verification"])
        test_cases = child_protection.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Child Protection" for tc in test_cases)
        assert all(
            tc.vulnerability_type == ChildProtectionType.AGE_VERIFICATION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        child_protection = ChildProtection(
            types=["data_privacy"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = child_protection.assess(
            model_callback=dummy_model_callback,
        )

        assert child_protection.is_vulnerable() is not None
        assert child_protection.simulated_attacks is not None and isinstance(
            child_protection.simulated_attacks, dict
        )
        assert child_protection.res is not None and isinstance(
            child_protection.res, dict
        )
        assert ChildProtectionType.DATA_PRIVACY in results
        assert len(results[ChildProtectionType.DATA_PRIVACY]) == 1
        test_case = results[ChildProtectionType.DATA_PRIVACY][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_child_protection_metric(self):
        from deepteam.metrics import ChildProtectionMetric

        child_protection = ChildProtection(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = child_protection._get_metric(ChildProtectionType.DATA_PRIVACY)
        assert isinstance(metric, ChildProtectionMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        child_protection = ChildProtection(
            types=["exposure_interaction"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await child_protection.a_assess(
            model_callback=dummy_model_callback,
        )

        assert child_protection.is_vulnerable() is not None
        assert child_protection.simulated_attacks is not None and isinstance(
            child_protection.simulated_attacks, dict
        )
        assert child_protection.res is not None and isinstance(
            child_protection.res, dict
        )
        assert ChildProtectionType.EXPOSURE_INTERACTION in results
        assert len(results[ChildProtectionType.EXPOSURE_INTERACTION]) == 1
        test_case = results[ChildProtectionType.EXPOSURE_INTERACTION][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
