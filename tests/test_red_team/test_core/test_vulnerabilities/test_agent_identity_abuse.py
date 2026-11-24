import pytest

from deepteam.vulnerabilities import AgentIdentityAbuse
from deepteam.vulnerabilities.agent_identity_abuse import AgentIdentityAbuseType
from deepteam.test_case import RTTestCase


class TestAgentIdentityAbuse:

    def test_agent_identity_abuse_all_types(self):
        types = [
            "agent_impersonation",
            "identity_inheritance",
            "cross_agent_trust_abuse",
        ]
        agent_identity_abuse = AgentIdentityAbuse(types=types)
        assert sorted(
            type.value for type in agent_identity_abuse.types
        ) == sorted(types)

    def test_agent_identity_abuse_all_types_default(self):
        agent_identity_abuse = AgentIdentityAbuse()
        assert sorted(
            type.value for type in agent_identity_abuse.types
        ) == sorted(type.value for type in AgentIdentityAbuseType)

    def test_agent_identity_abuse_agent_impersonation(self):
        types = ["agent_impersonation"]
        agent_identity_abuse = AgentIdentityAbuse(types=types)
        assert sorted(
            type.value for type in agent_identity_abuse.types
        ) == sorted(types)

    def test_agent_identity_abuse_identity_inheritance(self):
        types = ["identity_inheritance"]
        agent_identity_abuse = AgentIdentityAbuse(types=types)
        assert sorted(
            type.value for type in agent_identity_abuse.types
        ) == sorted(types)

    def test_agent_identity_abuse_cross_agent_trust_abuse(self):
        types = ["cross_agent_trust_abuse"]
        agent_identity_abuse = AgentIdentityAbuse(types=types)
        assert sorted(
            type.value for type in agent_identity_abuse.types
        ) == sorted(types)

    def test_agent_identity_abuse_all_types_invalid(self):
        types = [
            "agent_impersonation",
            "identity_inheritance",
            "cross_agent_trust_abuse",
            "invalid",
        ]
        with pytest.raises(ValueError):
            AgentIdentityAbuse(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        agent_identity_abuse = AgentIdentityAbuse(types=["agent_impersonation"])
        test_cases = agent_identity_abuse.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Agent Identity & Trust Abuse"
            for tc in test_cases
        )
        assert all(
            tc.vulnerability_type == AgentIdentityAbuseType.AGENT_IMPERSONATION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        agent_identity_abuse = AgentIdentityAbuse(
            types=["identity_inheritance"], async_mode=False
        )

        def dummy_model_callback(prompt):
            return prompt

        results = agent_identity_abuse.assess(
            model_callback=dummy_model_callback,
        )

        assert agent_identity_abuse.is_vulnerable() is not None
        assert (
            agent_identity_abuse.simulated_attacks is not None
            and isinstance(agent_identity_abuse.simulated_attacks, dict)
        )
        assert agent_identity_abuse.res is not None and isinstance(
            agent_identity_abuse.res, dict
        )
        assert AgentIdentityAbuseType.IDENTITY_INHERITANCE in results
        assert len(results[AgentIdentityAbuseType.IDENTITY_INHERITANCE]) == 1
        test_case = results[AgentIdentityAbuseType.IDENTITY_INHERITANCE][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_agent_identity_metric(self):
        from deepteam.metrics import AgentIdentityAbuseMetric

        agent_identity_abuse = AgentIdentityAbuse(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = agent_identity_abuse._get_metric(
            AgentIdentityAbuseType.IDENTITY_INHERITANCE
        )
        assert isinstance(metric, AgentIdentityAbuseMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        agent_identity_abuse = AgentIdentityAbuse(
            types=["cross_agent_trust_abuse"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await agent_identity_abuse.a_assess(
            model_callback=dummy_model_callback,
        )

        assert agent_identity_abuse.is_vulnerable() is not None
        assert (
            agent_identity_abuse.simulated_attacks is not None
            and isinstance(agent_identity_abuse.simulated_attacks, dict)
        )
        assert agent_identity_abuse.res is not None and isinstance(
            agent_identity_abuse.res, dict
        )
        assert AgentIdentityAbuseType.CROSS_AGENT_TRUST_ABUSE in results
        assert len(results[AgentIdentityAbuseType.CROSS_AGENT_TRUST_ABUSE]) == 1
        test_case = results[AgentIdentityAbuseType.CROSS_AGENT_TRUST_ABUSE][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
