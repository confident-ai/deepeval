import pytest

from deepteam.vulnerabilities import GoalTheft
from deepteam.vulnerabilities.goal_theft import GoalTheftType
from deepteam.test_case import RTTestCase


class TestGoalTheft:

    def test_goal_theft_all_types(self):
        types = [
            "escalating_probing",
            "cooperative_dialogue",
            "social_engineering",
        ]
        goal_theft = GoalTheft(types=types)
        assert sorted(type.value for type in goal_theft.types) == sorted(types)

    def test_goal_theft_all_types_default(self):
        goal_theft = GoalTheft()
        assert sorted(type.value for type in goal_theft.types) == sorted(
            type.value for type in GoalTheftType
        )

    def test_goal_theft_escalating_probing(self):
        types = ["escalating_probing"]
        goal_theft = GoalTheft(types=types)
        assert sorted(type.value for type in goal_theft.types) == sorted(types)

    def test_goal_theft_cooperative_dialogue(self):
        types = ["cooperative_dialogue"]
        goal_theft = GoalTheft(types=types)
        assert sorted(type.value for type in goal_theft.types) == sorted(types)

    def test_goal_theft_social_engineering(self):
        types = ["social_engineering"]
        goal_theft = GoalTheft(types=types)
        assert sorted(type.value for type in goal_theft.types) == sorted(types)

    def test_goal_theft_all_types_invalid(self):
        types = [
            "escalating_probing",
            "cooperative_dialogue",
            "social_engineering",
            "invalid",
        ]
        with pytest.raises(ValueError):
            GoalTheft(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        goal_theft = GoalTheft(types=["cooperative_dialogue"])
        test_cases = goal_theft.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Goal Theft" for tc in test_cases)
        assert all(
            tc.vulnerability_type == GoalTheftType.COOPERATIVE_DIALOGUE
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        goal_theft = GoalTheft(types=["cooperative_dialogue"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = goal_theft.assess(
            model_callback=dummy_model_callback,
        )

        assert goal_theft.is_vulnerable() is not None
        assert goal_theft.simulated_attacks is not None and isinstance(
            goal_theft.simulated_attacks, dict
        )
        assert goal_theft.res is not None and isinstance(goal_theft.res, dict)
        assert GoalTheftType.COOPERATIVE_DIALOGUE in results
        assert len(results[GoalTheftType.COOPERATIVE_DIALOGUE]) == 1
        test_case = results[GoalTheftType.COOPERATIVE_DIALOGUE][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_GoalTheft_metric(self):
        from deepteam.metrics.agentic import ExtractionSuccessMetric

        goal_theft = GoalTheft(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = goal_theft._get_metric(GoalTheftType.COOPERATIVE_DIALOGUE)
        assert isinstance(metric, ExtractionSuccessMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        goal_theft = GoalTheft(types=["cooperative_dialogue"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await goal_theft.a_assess(
            model_callback=dummy_model_callback,
        )

        assert goal_theft.is_vulnerable() is not None
        assert goal_theft.simulated_attacks is not None and isinstance(
            goal_theft.simulated_attacks, dict
        )
        assert goal_theft.res is not None and isinstance(goal_theft.res, dict)
        assert GoalTheftType.COOPERATIVE_DIALOGUE in results
        assert len(results[GoalTheftType.COOPERATIVE_DIALOGUE]) == 1
        test_case = results[GoalTheftType.COOPERATIVE_DIALOGUE][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
