import pytest

from deepteam.vulnerabilities import PersonalSafety
from deepteam.vulnerabilities.personal_safety import PersonalSafetyType
from deepteam.test_case import RTTestCase


class TestPersonalSafety:

    def test_personal_safety_all_types(self):
        types = [
            "bullying",
            "self_harm",
            "unsafe_practices",
            "dangerous_challenges",
            "stalking",
        ]
        personal_safety = PersonalSafety(types=types)
        assert sorted(type.value for type in personal_safety.types) == sorted(
            types
        )

    def test_personal_safety_all_types_default(self):
        personal_safety = PersonalSafety()
        assert sorted(type.value for type in personal_safety.types) == sorted(
            type.value for type in PersonalSafetyType
        )

    def test_personal_safety_bullying(self):
        types = ["bullying"]
        personal_safety = PersonalSafety(types=types)
        assert sorted(type.value for type in personal_safety.types) == sorted(
            types
        )

    def test_personal_safety_self_harm(self):
        types = ["self_harm"]
        personal_safety = PersonalSafety(types=types)
        assert sorted(type.value for type in personal_safety.types) == sorted(
            types
        )

    def test_personal_safety_unsafe_practices(self):
        types = ["unsafe_practices"]
        personal_safety = PersonalSafety(types=types)
        assert sorted(type.value for type in personal_safety.types) == sorted(
            types
        )

    def test_personal_safety_dangerous_challenges(self):
        types = ["dangerous_challenges"]
        personal_safety = PersonalSafety(types=types)
        assert sorted(type.value for type in personal_safety.types) == sorted(
            types
        )

    def test_personal_safety_stalking(self):
        types = ["stalking"]
        personal_safety = PersonalSafety(types=types)
        assert sorted(type.value for type in personal_safety.types) == sorted(
            types
        )

    def test_personal_safety_all_types_invalid(self):
        types = [
            "bullying",
            "self_harm",
            "unsafe_practices",
            "dangerous_challenges",
            "stalking",
            "invalid",
        ]
        with pytest.raises(ValueError):
            PersonalSafety(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        safety = PersonalSafety(types=["bullying"])
        test_cases = safety.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Personal Safety" for tc in test_cases)
        assert all(
            tc.vulnerability_type == PersonalSafetyType.BULLYING
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        safety = PersonalSafety(types=["bullying"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = safety.assess(
            model_callback=dummy_model_callback,
        )

        assert safety.is_vulnerable() is not None
        assert safety.simulated_attacks is not None and isinstance(
            safety.simulated_attacks, dict
        )
        assert safety.res is not None and isinstance(safety.res, dict)
        assert PersonalSafetyType.BULLYING in results
        assert len(results[PersonalSafetyType.BULLYING]) == 1
        test_case = results[PersonalSafetyType.BULLYING][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_PersonalSafety_metric(self):
        from deepteam.metrics import SafetyMetric

        safety = PersonalSafety(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = safety._get_metric(PersonalSafetyType.BULLYING)
        assert isinstance(metric, SafetyMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        safety = PersonalSafety(types=["bullying"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await safety.a_assess(
            model_callback=dummy_model_callback,
        )

        assert safety.is_vulnerable() is not None
        assert safety.simulated_attacks is not None and isinstance(
            safety.simulated_attacks, dict
        )
        assert safety.res is not None and isinstance(safety.res, dict)
        assert PersonalSafetyType.BULLYING in results
        assert len(results[PersonalSafetyType.BULLYING]) == 1
        test_case = results[PersonalSafetyType.BULLYING][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
