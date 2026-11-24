import pytest

from deepteam.vulnerabilities import IllegalActivity
from deepteam.vulnerabilities.illegal_activity import IllegalActivityType
from deepteam.test_case import RTTestCase


class TestIllegalActivity:

    def test_illegal_activity_all_types(self):
        types = [
            "weapons",
            "illegal_drugs",
            "violent_crimes",
            "non_violent_crimes",
            "sex_crimes",
            "cybercrime",
            "child_exploitation",
        ]
        illegal_activity = IllegalActivity(types=types)
        assert sorted(type.value for type in illegal_activity.types) == sorted(
            types
        )

    def test_illegal_activity_all_types_default(self):
        illegal_activity = IllegalActivity()
        assert sorted(type.value for type in illegal_activity.types) == sorted(
            type.value for type in IllegalActivityType
        )

    def test_illegal_activity_weapons(self):
        types = ["weapons"]
        illegal_activity = IllegalActivity(types=types)
        assert sorted(type.value for type in illegal_activity.types) == sorted(
            types
        )

    def test_illegal_activity_illegal_drugs(self):
        types = ["illegal_drugs"]
        illegal_activity = IllegalActivity(types=types)
        assert sorted(type.value for type in illegal_activity.types) == sorted(
            types
        )

    def test_illegal_activity_violent_crimes(self):
        types = ["violent_crimes"]
        illegal_activity = IllegalActivity(types=types)
        assert sorted(type.value for type in illegal_activity.types) == sorted(
            types
        )

    def test_illegal_activity_non_violent_crimes(self):
        types = ["non_violent_crimes"]
        illegal_activity = IllegalActivity(types=types)
        assert sorted(type.value for type in illegal_activity.types) == sorted(
            types
        )

    def test_illegal_activity_sex_crimes(self):
        types = ["sex_crimes"]
        illegal_activity = IllegalActivity(types=types)
        assert sorted(type.value for type in illegal_activity.types) == sorted(
            types
        )

    def test_illegal_activity_cybercrime(self):
        types = ["cybercrime"]
        illegal_activity = IllegalActivity(types=types)
        assert sorted(type.value for type in illegal_activity.types) == sorted(
            types
        )

    def test_illegal_activity_child_exploitation(self):
        types = ["child_exploitation"]
        illegal_activity = IllegalActivity(types=types)
        assert sorted(type.value for type in illegal_activity.types) == sorted(
            types
        )

    def test_illegal_activity_all_types_invalid(self):
        types = [
            "weapons",
            "illegal_drugs",
            "violent_crimes",
            "non_violent_crimes",
            "sex_crimes",
            "cybercrime",
            "child_exploitation",
            "invalid",
        ]
        with pytest.raises(ValueError):
            IllegalActivity(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        illegal = IllegalActivity(types=["weapons"])
        test_cases = illegal.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Illegal Activity" for tc in test_cases)
        assert all(
            tc.vulnerability_type == IllegalActivityType.WEAPONS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        illegal = IllegalActivity(types=["weapons"], async_mode=False)

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = illegal.assess(
            model_callback=dummy_model_callback,
        )

        assert illegal.is_vulnerable() is not None
        assert illegal.simulated_attacks is not None and isinstance(
            illegal.simulated_attacks, dict
        )
        assert illegal.res is not None and isinstance(illegal.res, dict)
        assert IllegalActivityType.WEAPONS in results
        assert len(results[IllegalActivityType.WEAPONS]) == 1
        test_case = results[IllegalActivityType.WEAPONS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_IllegalActivity_metric(self):
        from deepteam.metrics import IllegalMetric

        illegal = IllegalActivity(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = illegal._get_metric(IllegalActivityType.WEAPONS)
        assert isinstance(metric, IllegalMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        illegal = IllegalActivity(types=["weapons"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await illegal.a_assess(
            model_callback=dummy_model_callback,
        )

        assert illegal.is_vulnerable() is not None
        assert illegal.simulated_attacks is not None and isinstance(
            illegal.simulated_attacks, dict
        )
        assert illegal.res is not None and isinstance(illegal.res, dict)
        assert IllegalActivityType.WEAPONS in results
        assert len(results[IllegalActivityType.WEAPONS]) == 1
        test_case = results[IllegalActivityType.WEAPONS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
