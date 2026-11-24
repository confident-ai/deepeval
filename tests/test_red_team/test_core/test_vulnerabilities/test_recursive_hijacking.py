import pytest

from deepteam.vulnerabilities import RecursiveHijacking
from deepteam.vulnerabilities.recursive_hijacking import (
    RecursiveHijackingType,
)
from deepteam.test_case import RTTestCase


class TestRecursiveHijacking:

    def test_recursive_hijacking_all_types(self):
        types = [
            "self_modifying_goals",
            "recursive_objective_chaining",
            "goal_propagation_attacks",
        ]
        recursive_hijacking = RecursiveHijacking(types=types)
        assert sorted(
            type.value for type in recursive_hijacking.types
        ) == sorted(types)

    def test_recursive_hijacking_all_types_default(self):
        recursive_hijacking = RecursiveHijacking()
        assert sorted(
            type.value for type in recursive_hijacking.types
        ) == sorted(type.value for type in RecursiveHijackingType)

    def test_recursive_hijacking_self_modifying_goals(self):
        types = ["self_modifying_goals"]
        recursive_hijacking = RecursiveHijacking(types=types)
        assert sorted(
            type.value for type in recursive_hijacking.types
        ) == sorted(types)

    def test_recursive_hijacking_recursive_objective_chaining(self):
        types = ["recursive_objective_chaining"]
        recursive_hijacking = RecursiveHijacking(types=types)
        assert sorted(
            type.value for type in recursive_hijacking.types
        ) == sorted(types)

    def test_recursive_hijacking_goal_propagation_attacks(self):
        types = ["goal_propagation_attacks"]
        recursive_hijacking = RecursiveHijacking(types=types)
        assert sorted(
            type.value for type in recursive_hijacking.types
        ) == sorted(types)

    def test_recursive_hijacking_all_types_invalid(self):
        types = [
            "self_modifying_goals",
            "recursive_objective_chaining",
            "goal_propagation_attacks",
            "invalid",
        ]
        with pytest.raises(ValueError):
            RecursiveHijacking(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        recursive_hijack = RecursiveHijacking(
            types=["goal_propagation_attacks"]
        )
        test_cases = recursive_hijack.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Recursive Hijacking" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type
            == RecursiveHijackingType.GOAL_PROPAGATION_ATTACKS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        recursive_hijack = RecursiveHijacking(
            types=["goal_propagation_attacks"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = recursive_hijack.assess(
            model_callback=dummy_model_callback,
        )

        assert recursive_hijack.is_vulnerable() is not None
        assert recursive_hijack.simulated_attacks is not None and isinstance(
            recursive_hijack.simulated_attacks, dict
        )
        assert recursive_hijack.res is not None and isinstance(
            recursive_hijack.res, dict
        )
        assert RecursiveHijackingType.GOAL_PROPAGATION_ATTACKS in results
        assert (
            len(results[RecursiveHijackingType.GOAL_PROPAGATION_ATTACKS]) == 1
        )
        test_case = results[RecursiveHijackingType.GOAL_PROPAGATION_ATTACKS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_RecursiveHijacking_metric(self):
        from deepteam.metrics.agentic import SubversionSuccessMetric

        recursive_hijack = RecursiveHijacking(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = recursive_hijack._get_metric(
            RecursiveHijackingType.GOAL_PROPAGATION_ATTACKS
        )
        assert isinstance(metric, SubversionSuccessMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        recursive_hijack = RecursiveHijacking(
            types=["goal_propagation_attacks"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await recursive_hijack.a_assess(
            model_callback=dummy_model_callback,
        )

        assert recursive_hijack.is_vulnerable() is not None
        assert recursive_hijack.simulated_attacks is not None and isinstance(
            recursive_hijack.simulated_attacks, dict
        )
        assert recursive_hijack.res is not None and isinstance(
            recursive_hijack.res, dict
        )
        assert RecursiveHijackingType.GOAL_PROPAGATION_ATTACKS in results
        assert (
            len(results[RecursiveHijackingType.GOAL_PROPAGATION_ATTACKS]) == 1
        )
        test_case = results[RecursiveHijackingType.GOAL_PROPAGATION_ATTACKS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
