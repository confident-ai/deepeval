import pytest

from deepteam.vulnerabilities import DebugAccess
from deepteam.vulnerabilities.debug_access import DebugAccessType
from deepteam.test_case import RTTestCase


class TestDebugAccess:

    def test_debug_access_all_types(self):
        types = [
            "debug_mode_bypass",
            "development_endpoint_access",
            "administrative_interface_exposure",
        ]
        debug_access = DebugAccess(types=types)
        assert sorted(type.value for type in debug_access.types) == sorted(
            types
        )

    def test_debug_access_all_types_default(self):
        debug_access = DebugAccess()
        assert sorted(type.value for type in debug_access.types) == sorted(
            type.value for type in DebugAccessType
        )

    def test_debug_access_debug_mode_bypass(self):
        types = ["debug_mode_bypass"]
        debug_access = DebugAccess(types=types)
        assert sorted(type.value for type in debug_access.types) == sorted(
            types
        )

    def test_debug_access_development_endpoint_access(self):
        types = ["development_endpoint_access"]
        debug_access = DebugAccess(types=types)
        assert sorted(type.value for type in debug_access.types) == sorted(
            types
        )

    def test_debug_access_administrative_interface_exposure(self):
        types = ["administrative_interface_exposure"]
        debug_access = DebugAccess(types=types)
        assert sorted(type.value for type in debug_access.types) == sorted(
            types
        )

    def test_debug_access_all_types_invalid(self):
        types = [
            "debug_mode_bypass",
            "development_endpoint_access",
            "administrative_interface_exposure",
            "invalid",
        ]
        with pytest.raises(ValueError):
            DebugAccess(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        debug_access = DebugAccess(types=["debug_mode_bypass"])
        test_cases = debug_access.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Debug Access" for tc in test_cases)
        assert all(
            tc.vulnerability_type == DebugAccessType.DEBUG_MODE_BYPASS
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        debug_access = DebugAccess(
            types=["debug_mode_bypass"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = debug_access.assess(
            model_callback=dummy_model_callback,
        )

        assert debug_access.is_vulnerable() is not None
        assert debug_access.simulated_attacks is not None and isinstance(
            debug_access.simulated_attacks, dict
        )
        assert debug_access.res is not None and isinstance(
            debug_access.res, dict
        )
        assert DebugAccessType.DEBUG_MODE_BYPASS in results
        assert len(results[DebugAccessType.DEBUG_MODE_BYPASS]) == 1
        test_case = results[DebugAccessType.DEBUG_MODE_BYPASS][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_DebugAccess_metric(self):
        from deepteam.metrics import DebugAccessMetric

        debug_access = DebugAccess(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = debug_access._get_metric(DebugAccessType.DEBUG_MODE_BYPASS)
        assert isinstance(metric, DebugAccessMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        debug_access = DebugAccess(types=["debug_mode_bypass"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await debug_access.a_assess(
            model_callback=dummy_model_callback,
        )

        assert debug_access.is_vulnerable() is not None
        assert debug_access.simulated_attacks is not None and isinstance(
            debug_access.simulated_attacks, dict
        )
        assert debug_access.res is not None and isinstance(
            debug_access.res, dict
        )
        assert DebugAccessType.DEBUG_MODE_BYPASS in results
        assert len(results[DebugAccessType.DEBUG_MODE_BYPASS]) == 1
        test_case = results[DebugAccessType.DEBUG_MODE_BYPASS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
