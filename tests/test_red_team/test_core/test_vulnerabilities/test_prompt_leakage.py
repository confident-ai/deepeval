import pytest

from deepteam.vulnerabilities import PromptLeakage
from deepteam.vulnerabilities.prompt_leakage import PromptLeakageType
from deepteam.test_case import RTTestCase


class TestPromptLeakage:

    def test_prompt_leakage_all_types(self):
        types = [
            "secrets_and_credentials",
            "instructions",
            "guard_exposure",
            "permissions_and_roles",
        ]
        prompt_leakage = PromptLeakage(types=types)
        assert sorted(type.value for type in prompt_leakage.types) == sorted(
            types
        )

    def test_prompt_leakage_all_types_default(self):
        prompt_leakage = PromptLeakage()
        assert sorted(type.value for type in prompt_leakage.types) == sorted(
            type.value for type in PromptLeakageType
        )

    def test_prompt_leakage_secrets_and_credentials(self):
        types = ["secrets_and_credentials"]
        prompt_leakage = PromptLeakage(types=types)
        assert sorted(type.value for type in prompt_leakage.types) == sorted(
            types
        )

    def test_prompt_leakage_instructions(self):
        types = ["instructions"]
        prompt_leakage = PromptLeakage(types=types)
        assert sorted(type.value for type in prompt_leakage.types) == sorted(
            types
        )

    def test_prompt_leakage_guard_exposure(self):
        types = ["guard_exposure"]
        prompt_leakage = PromptLeakage(types=types)
        assert sorted(type.value for type in prompt_leakage.types) == sorted(
            types
        )

    def test_prompt_leakage_permissions_and_roles(self):
        types = ["permissions_and_roles"]
        prompt_leakage = PromptLeakage(types=types)
        assert sorted(type.value for type in prompt_leakage.types) == sorted(
            types
        )

    def test_prompt_leakage_all_types_invalid(self):
        types = [
            "secrets_and_credentials",
            "instructions",
            "guard_exposure",
            "permissions_and _oles",
            "invalid",
        ]
        with pytest.raises(ValueError):
            PromptLeakage(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        prompt_leakage = PromptLeakage(types=["secrets_and_credentials"])
        test_cases = prompt_leakage.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Prompt Leakage" for tc in test_cases)
        assert all(
            tc.vulnerability_type == PromptLeakageType.SECRETS_AND_CREDENTIALS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        prompt_leakage = PromptLeakage(
            types=["secrets_and_credentials"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = prompt_leakage.assess(
            model_callback=dummy_model_callback,
        )

        assert prompt_leakage.is_vulnerable() is not None
        assert prompt_leakage.simulated_attacks is not None and isinstance(
            prompt_leakage.simulated_attacks, dict
        )
        assert prompt_leakage.res is not None and isinstance(
            prompt_leakage.res, dict
        )
        assert PromptLeakageType.SECRETS_AND_CREDENTIALS in results
        assert len(results[PromptLeakageType.SECRETS_AND_CREDENTIALS]) == 1
        test_case = results[PromptLeakageType.SECRETS_AND_CREDENTIALS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_PromptLeakage_metric(self):
        from deepteam.metrics import PromptExtractionMetric

        prompt_leakage = PromptLeakage(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = prompt_leakage._get_metric(
            PromptLeakageType.SECRETS_AND_CREDENTIALS
        )
        assert isinstance(metric, PromptExtractionMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        prompt_leakage = PromptLeakage(
            types=["secrets_and_credentials"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await prompt_leakage.a_assess(
            model_callback=dummy_model_callback,
        )

        assert prompt_leakage.is_vulnerable() is not None
        assert prompt_leakage.simulated_attacks is not None and isinstance(
            prompt_leakage.simulated_attacks, dict
        )
        assert prompt_leakage.res is not None and isinstance(
            prompt_leakage.res, dict
        )
        assert PromptLeakageType.SECRETS_AND_CREDENTIALS in results
        assert len(results[PromptLeakageType.SECRETS_AND_CREDENTIALS]) == 1
        test_case = results[PromptLeakageType.SECRETS_AND_CREDENTIALS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
