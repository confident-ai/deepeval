import pytest

from deepteam.vulnerabilities import ToolMetadataPoisoning
from deepteam.vulnerabilities.tool_metadata_poisoning import (
    ToolMetadataPoisoningType,
)
from deepteam.test_case import RTTestCase


class TestToolMetadataPoisoning:

    def test_tool_metadata_poisoning_all_types(self):
        types = [
            "schema_manipulation",
            "description_deception",
            "permission_misrepresentation",
            "registry_poisoning",
        ]
        tool_metadata_poisoning = ToolMetadataPoisoning(types=types)
        assert sorted(
            type.value for type in tool_metadata_poisoning.types
        ) == sorted(types)

    def test_tool_metadata_poisoning_all_types_default(self):
        tool_metadata_poisoning = ToolMetadataPoisoning()
        assert sorted(
            type.value for type in tool_metadata_poisoning.types
        ) == sorted(type.value for type in ToolMetadataPoisoningType)

    def test_tool_metadata_poisoning_schema_manipulation(self):
        types = ["schema_manipulation"]
        tool_metadata_poisoning = ToolMetadataPoisoning(types=types)
        assert sorted(
            type.value for type in tool_metadata_poisoning.types
        ) == sorted(types)

    def test_tool_metadata_poisoning_description_deception(self):
        types = ["description_deception"]
        tool_metadata_poisoning = ToolMetadataPoisoning(types=types)
        assert sorted(
            type.value for type in tool_metadata_poisoning.types
        ) == sorted(types)

    def test_tool_metadata_poisoning_permission_misrepresentation(self):
        types = ["permission_misrepresentation"]
        tool_metadata_poisoning = ToolMetadataPoisoning(types=types)
        assert sorted(
            type.value for type in tool_metadata_poisoning.types
        ) == sorted(types)

    def test_tool_metadata_poisoning_registry_poisoning(self):
        types = ["registry_poisoning"]
        tool_metadata_poisoning = ToolMetadataPoisoning(types=types)
        assert sorted(
            type.value for type in tool_metadata_poisoning.types
        ) == sorted(types)

    def test_tool_metadata_poisoning_all_types_invalid(self):
        types = [
            "schema_manipulation",
            "description_deception",
            "permission_misrepresentation",
            "registry_poisoning",
            "invalid",
        ]
        with pytest.raises(ValueError):
            ToolMetadataPoisoning(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        tool_metadata_poisoning = ToolMetadataPoisoning(
            types=["schema_manipulation"]
        )
        test_cases = tool_metadata_poisoning.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Tool Metadata Poisoning" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type
            == ToolMetadataPoisoningType.SCHEMA_MANIPULATION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        tool_metadata_poisoning = ToolMetadataPoisoning(
            types=["description_deception"], async_mode=False
        )

        def dummy_model_callback(prompt):
            return prompt

        results = tool_metadata_poisoning.assess(
            model_callback=dummy_model_callback,
        )

        assert tool_metadata_poisoning.is_vulnerable() is not None
        assert (
            tool_metadata_poisoning.simulated_attacks is not None
            and isinstance(tool_metadata_poisoning.simulated_attacks, dict)
        )
        assert tool_metadata_poisoning.res is not None and isinstance(
            tool_metadata_poisoning.res, dict
        )
        assert ToolMetadataPoisoningType.DESCRIPTION_DECEPTION in results
        assert (
            len(results[ToolMetadataPoisoningType.DESCRIPTION_DECEPTION]) == 1
        )
        test_case = results[ToolMetadataPoisoningType.DESCRIPTION_DECEPTION][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_tool_metadata_poisoning_metric(self):
        from deepteam.metrics import ToolMetadataPoisoningMetric

        tool_metadata_poisoning = ToolMetadataPoisoning(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = tool_metadata_poisoning._get_metric(
            ToolMetadataPoisoningType.DESCRIPTION_DECEPTION
        )
        assert isinstance(metric, ToolMetadataPoisoningMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        tool_metadata_poisoning = ToolMetadataPoisoning(
            types=["registry_poisoning"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await tool_metadata_poisoning.a_assess(
            model_callback=dummy_model_callback,
        )

        assert tool_metadata_poisoning.is_vulnerable() is not None
        assert (
            tool_metadata_poisoning.simulated_attacks is not None
            and isinstance(tool_metadata_poisoning.simulated_attacks, dict)
        )
        assert tool_metadata_poisoning.res is not None and isinstance(
            tool_metadata_poisoning.res, dict
        )
        assert ToolMetadataPoisoningType.REGISTRY_POISONING in results
        assert len(results[ToolMetadataPoisoningType.REGISTRY_POISONING]) == 1
        test_case = results[ToolMetadataPoisoningType.REGISTRY_POISONING][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
