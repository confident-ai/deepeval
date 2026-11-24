import pytest

from deepteam.vulnerabilities import ExternalSystemAbuse
from deepteam.vulnerabilities.external_system_abuse import (
    ExternalSystemAbuseType,
)
from deepteam.test_case import RTTestCase


class TestExternalSystemAbuse:

    def test_external_system_abuse_all_types(self):
        types = [
            "data_exfiltration",
            "communications_spam",
            "internal_spoofing",
        ]
        external_system_abuse = ExternalSystemAbuse(types=types)
        assert sorted(
            type.value for type in external_system_abuse.types
        ) == sorted(types)

    def test_external_system_abuse_all_types_default(self):
        external_system_abuse = ExternalSystemAbuse()
        assert sorted(
            type.value for type in external_system_abuse.types
        ) == sorted(type.value for type in ExternalSystemAbuseType)

    def test_external_system_abuse_data_exfiltration(self):
        types = ["data_exfiltration"]
        external_system_abuse = ExternalSystemAbuse(types=types)
        assert sorted(
            type.value for type in external_system_abuse.types
        ) == sorted(types)

    def test_external_system_abuse_communications_spam(self):
        types = ["communications_spam"]
        external_system_abuse = ExternalSystemAbuse(types=types)
        assert sorted(
            type.value for type in external_system_abuse.types
        ) == sorted(types)

    def test_external_system_abuse_internal_spoofing(self):
        types = ["internal_spoofing"]
        external_system_abuse = ExternalSystemAbuse(types=types)
        assert sorted(
            type.value for type in external_system_abuse.types
        ) == sorted(types)

    def test_external_system_abuse_all_types_invalid(self):
        types = [
            "data_exfiltration",
            "communications_spam",
            "internal_spoofing",
            "invalid",
        ]
        with pytest.raises(ValueError):
            ExternalSystemAbuse(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        external_system_abuse = ExternalSystemAbuse(types=["data_exfiltration"])
        test_cases = external_system_abuse.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "External System Abuse" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type == ExternalSystemAbuseType.DATA_EXFILTRATION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        external_system_abuse = ExternalSystemAbuse(
            types=["communications_spam"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = external_system_abuse.assess(
            model_callback=dummy_model_callback,
        )

        assert external_system_abuse.is_vulnerable() is not None
        assert (
            external_system_abuse.simulated_attacks is not None
            and isinstance(external_system_abuse.simulated_attacks, dict)
        )
        assert external_system_abuse.res is not None and isinstance(
            external_system_abuse.res, dict
        )
        assert ExternalSystemAbuseType.COMMUNICATIONS_SPAM in results
        assert len(results[ExternalSystemAbuseType.COMMUNICATIONS_SPAM]) == 1
        test_case = results[ExternalSystemAbuseType.COMMUNICATIONS_SPAM][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_cross_context_retrieval_metric(self):
        from deepteam.metrics import ExternalSystemAbuseMetric

        external_system_abuse = ExternalSystemAbuse(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = external_system_abuse._get_metric(
            ExternalSystemAbuseType.INTERNAL_SPOOFING
        )
        assert isinstance(metric, ExternalSystemAbuseMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        external_system_abuse = ExternalSystemAbuse(
            types=["internal_spoofing"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await external_system_abuse.a_assess(
            model_callback=dummy_model_callback,
        )

        assert external_system_abuse.is_vulnerable() is not None
        assert (
            external_system_abuse.simulated_attacks is not None
            and isinstance(external_system_abuse.simulated_attacks, dict)
        )
        assert external_system_abuse.res is not None and isinstance(
            external_system_abuse.res, dict
        )
        assert ExternalSystemAbuseType.INTERNAL_SPOOFING in results
        assert len(results[ExternalSystemAbuseType.INTERNAL_SPOOFING]) == 1
        test_case = results[ExternalSystemAbuseType.INTERNAL_SPOOFING][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
