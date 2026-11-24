import pytest

from deepteam.vulnerabilities import InsecureInterAgentCommunication
from deepteam.vulnerabilities.insecure_inter_agent_communication import (
    InsecureInterAgentCommunicationType,
)
from deepteam.test_case import RTTestCase


class TestInsecureInterAgentCommunication:

    def test_insecure_interagent_communication_all_types(self):
        types = [
            "message_spoofing",
            "message_injection",
            "agent_in_the_middle",
        ]
        insecure_interagent_communication = InsecureInterAgentCommunication(
            types=types
        )
        assert sorted(
            type.value for type in insecure_interagent_communication.types
        ) == sorted(types)

    def test_insecure_interagent_communication_all_types_default(self):
        insecure_interagent_communication = InsecureInterAgentCommunication()
        assert sorted(
            type.value for type in insecure_interagent_communication.types
        ) == sorted(type.value for type in InsecureInterAgentCommunicationType)

    def test_insecure_interagent_communication_message_spoofing(self):
        types = ["message_spoofing"]
        insecure_interagent_communication = InsecureInterAgentCommunication(
            types=types
        )
        assert sorted(
            type.value for type in insecure_interagent_communication.types
        ) == sorted(types)

    def test_insecure_interagent_communication_message_injection(self):
        types = ["message_injection"]
        insecure_interagent_communication = InsecureInterAgentCommunication(
            types=types
        )
        assert sorted(
            type.value for type in insecure_interagent_communication.types
        ) == sorted(types)

    def test_insecure_interagent_communication_agent_in_the_middle(self):
        types = ["agent_in_the_middle"]
        insecure_interagent_communication = InsecureInterAgentCommunication(
            types=types
        )
        assert sorted(
            type.value for type in insecure_interagent_communication.types
        ) == sorted(types)

    def test_insecure_interagent_communication_all_types_invalid(self):
        types = [
            "message_spoofing",
            "message_injection",
            "agent_in_the_middle",
            "invalid",
        ]
        with pytest.raises(ValueError):
            InsecureInterAgentCommunication(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        insecure_interagent_communication = InsecureInterAgentCommunication(
            types=["message_spoofing"]
        )
        test_cases = insecure_interagent_communication.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Inter-Agent Communication Compromise"
            for tc in test_cases
        )
        assert all(
            tc.vulnerability_type
            == InsecureInterAgentCommunicationType.MESSAGE_SPOOFING
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        insecure_interagent_communication = InsecureInterAgentCommunication(
            types=["message_injection"], async_mode=False
        )

        def dummy_model_callback(prompt):
            return prompt

        results = insecure_interagent_communication.assess(
            model_callback=dummy_model_callback,
        )

        assert insecure_interagent_communication.is_vulnerable() is not None
        assert (
            insecure_interagent_communication.simulated_attacks is not None
            and isinstance(
                insecure_interagent_communication.simulated_attacks, dict
            )
        )
        assert insecure_interagent_communication.res is not None and isinstance(
            insecure_interagent_communication.res, dict
        )
        assert InsecureInterAgentCommunicationType.MESSAGE_INJECTION in results
        assert (
            len(results[InsecureInterAgentCommunicationType.MESSAGE_INJECTION])
            == 1
        )
        test_case = results[
            InsecureInterAgentCommunicationType.MESSAGE_INJECTION
        ][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_insecure_interagent_communication_metric(self):
        from deepteam.metrics import InsecureInterAgentCommunicationMetric

        insecure_interagent_communication = InsecureInterAgentCommunication(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = insecure_interagent_communication._get_metric(
            InsecureInterAgentCommunicationType.MESSAGE_INJECTION
        )
        assert isinstance(metric, InsecureInterAgentCommunicationMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        insecure_interagent_communication = InsecureInterAgentCommunication(
            types=["agent_in_the_middle"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await insecure_interagent_communication.a_assess(
            model_callback=dummy_model_callback,
        )

        assert insecure_interagent_communication.is_vulnerable() is not None
        assert (
            insecure_interagent_communication.simulated_attacks is not None
            and isinstance(
                insecure_interagent_communication.simulated_attacks, dict
            )
        )
        assert insecure_interagent_communication.res is not None and isinstance(
            insecure_interagent_communication.res, dict
        )
        assert (
            InsecureInterAgentCommunicationType.AGENT_IN_THE_MIDDLE in results
        )
        assert (
            len(
                results[InsecureInterAgentCommunicationType.AGENT_IN_THE_MIDDLE]
            )
            == 1
        )
        test_case = results[
            InsecureInterAgentCommunicationType.AGENT_IN_THE_MIDDLE
        ][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
