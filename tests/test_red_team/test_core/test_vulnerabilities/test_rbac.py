import pytest

from deepteam.vulnerabilities import RBAC
from deepteam.vulnerabilities.rbac import RBACType
from deepteam.test_case import RTTestCase


class TestRBAC:

    def test_rbac_all_types(self):
        types = [
            "role_bypass",
            "privilege_escalation",
            "unauthorized_role_assumption",
        ]
        rbac = RBAC(types=types)
        assert sorted(type.value for type in rbac.types) == sorted(types)

    def test_rbac_all_types_default(self):
        rbac = RBAC()
        assert sorted(type.value for type in rbac.types) == sorted(
            type.value for type in RBACType
        )

    def test_rbac_role_bypass(self):
        types = ["role_bypass"]
        rbac = RBAC(types=types)
        assert sorted(type.value for type in rbac.types) == sorted(types)

    def test_rbac_privilege_escalation(self):
        types = ["privilege_escalation"]
        rbac = RBAC(types=types)
        assert sorted(type.value for type in rbac.types) == sorted(types)

    def test_rbac_unauthorized_role_assumption(self):
        types = ["unauthorized_role_assumption"]
        rbac = RBAC(types=types)
        assert sorted(type.value for type in rbac.types) == sorted(types)

    def test_rbac_all_types_invalid(self):
        types = [
            "role_bypass",
            "privilege_escalation",
            "unauthorized_role_assumption",
            "invalid",
        ]
        with pytest.raises(ValueError):
            RBAC(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        rbac = RBAC(types=["role_bypass"])
        test_cases = rbac.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "RBAC" for tc in test_cases)
        assert all(
            tc.vulnerability_type == RBACType.ROLE_BYPASS for tc in test_cases
        )

    def test_assess_returns_results(self):
        rbac = RBAC(types=["role_bypass"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = rbac.assess(
            model_callback=dummy_model_callback,
        )

        assert rbac.is_vulnerable() is not None
        assert rbac.simulated_attacks is not None and isinstance(
            rbac.simulated_attacks, dict
        )
        assert rbac.res is not None and isinstance(rbac.res, dict)
        assert RBACType.ROLE_BYPASS in results
        assert len(results[RBACType.ROLE_BYPASS]) == 1
        test_case = results[RBACType.ROLE_BYPASS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_RBAC_metric(self):
        from deepteam.metrics import RBACMetric

        rbac = RBAC(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = rbac._get_metric(RBACType.ROLE_BYPASS)
        assert isinstance(metric, RBACMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        rbac = RBAC(types=["role_bypass"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await rbac.a_assess(
            model_callback=dummy_model_callback,
        )

        assert rbac.is_vulnerable() is not None
        assert rbac.simulated_attacks is not None and isinstance(
            rbac.simulated_attacks, dict
        )
        assert rbac.res is not None and isinstance(rbac.res, dict)
        assert RBACType.ROLE_BYPASS in results
        assert len(results[RBACType.ROLE_BYPASS]) == 1
        test_case = results[RBACType.ROLE_BYPASS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
