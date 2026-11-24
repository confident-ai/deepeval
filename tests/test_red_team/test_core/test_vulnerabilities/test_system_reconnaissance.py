import pytest

from deepteam.vulnerabilities import SystemReconnaissance
from deepteam.vulnerabilities.system_reconnaissance import (
    SystemReconnaissanceType,
)
from deepteam.test_case import RTTestCase


class TestSystemReconnaissance:

    def test_system_reconnaissance_all_types(self):
        types = ["file_metadata", "database_schema", "retrieval_config"]
        system_reconnaissance = SystemReconnaissance(types=types)
        assert sorted(
            type.value for type in system_reconnaissance.types
        ) == sorted(types)

    def test_system_reconnaissance_all_types_default(self):
        system_reconnaissance = SystemReconnaissance()
        assert sorted(
            type.value for type in system_reconnaissance.types
        ) == sorted(type.value for type in SystemReconnaissanceType)

    def test_system_reconnaissance_file_metadata(self):
        types = ["file_metadata"]
        system_reconnaissance = SystemReconnaissance(types=types)
        assert sorted(
            type.value for type in system_reconnaissance.types
        ) == sorted(types)

    def test_system_reconnaissance_database_schema(self):
        types = ["database_schema"]
        system_reconnaissance = SystemReconnaissance(types=types)
        assert sorted(
            type.value for type in system_reconnaissance.types
        ) == sorted(types)

    def test_system_reconnaissance_retrieval_config(self):
        types = ["retrieval_config"]
        system_reconnaissance = SystemReconnaissance(types=types)
        assert sorted(
            type.value for type in system_reconnaissance.types
        ) == sorted(types)

    def test_system_reconnaissance_all_types_invalid(self):
        types = [
            "file_metadata",
            "database_schema",
            "retrieval_config",
            "invalid",
        ]
        with pytest.raises(ValueError):
            SystemReconnaissance(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        system_reconnaissance = SystemReconnaissance(types=["file_metadata"])
        test_cases = system_reconnaissance.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "System Reconnaissance" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type == SystemReconnaissanceType.FILE_METADATA
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        system_reconnaissance = SystemReconnaissance(
            types=["database_schema"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = system_reconnaissance.assess(
            model_callback=dummy_model_callback,
        )

        assert system_reconnaissance.is_vulnerable() is not None
        assert (
            system_reconnaissance.simulated_attacks is not None
            and isinstance(system_reconnaissance.simulated_attacks, dict)
        )
        assert system_reconnaissance.res is not None and isinstance(
            system_reconnaissance.res, dict
        )
        assert SystemReconnaissanceType.DATABASE_SCHEMA in results
        assert len(results[SystemReconnaissanceType.DATABASE_SCHEMA]) == 1
        test_case = results[SystemReconnaissanceType.DATABASE_SCHEMA][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_cross_context_retrieval_metric(self):
        from deepteam.metrics import SystemReconnaissanceMetric

        system_reconnaissance = SystemReconnaissance(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = system_reconnaissance._get_metric(
            SystemReconnaissanceType.FILE_METADATA
        )
        assert isinstance(metric, SystemReconnaissanceMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        system_reconnaissance = SystemReconnaissance(
            types=["retrieval_config"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await system_reconnaissance.a_assess(
            model_callback=dummy_model_callback,
        )

        assert system_reconnaissance.is_vulnerable() is not None
        assert (
            system_reconnaissance.simulated_attacks is not None
            and isinstance(system_reconnaissance.simulated_attacks, dict)
        )
        assert system_reconnaissance.res is not None and isinstance(
            system_reconnaissance.res, dict
        )
        assert SystemReconnaissanceType.RETRIEVAL_CONFIG in results
        assert len(results[SystemReconnaissanceType.RETRIEVAL_CONFIG]) == 1
        test_case = results[SystemReconnaissanceType.RETRIEVAL_CONFIG][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
