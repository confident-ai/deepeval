import pytest

from deepteam.vulnerabilities import UnexpectedCodeExecution
from deepteam.vulnerabilities.unexpected_code_execution import (
    UnexpectedCodeExecutionType,
)
from deepteam.test_case import RTTestCase


class TestUnexpectedCodeExecution:

    def test_unexpected_code_execution_all_types(self):
        types = [
            "unauthorized_code_execution",
            "shell_command_execution",
            "eval_usage",
        ]
        unexpected_code_execution = UnexpectedCodeExecution(types=types)
        assert sorted(
            type.value for type in unexpected_code_execution.types
        ) == sorted(types)

    def test_unexpected_code_execution_all_types_default(self):
        unexpected_code_execution = UnexpectedCodeExecution()
        assert sorted(
            type.value for type in unexpected_code_execution.types
        ) == sorted(type.value for type in UnexpectedCodeExecutionType)

    def test_unexpected_code_execution_unauthorized_code_execution(self):
        types = ["unauthorized_code_execution"]
        unexpected_code_execution = UnexpectedCodeExecution(types=types)
        assert sorted(
            type.value for type in unexpected_code_execution.types
        ) == sorted(types)

    def test_unexpected_code_execution_shell_command_execution(self):
        types = ["shell_command_execution"]
        unexpected_code_execution = UnexpectedCodeExecution(types=types)
        assert sorted(
            type.value for type in unexpected_code_execution.types
        ) == sorted(types)

    def test_unexpected_code_execution_eval_usage(self):
        types = ["eval_usage"]
        unexpected_code_execution = UnexpectedCodeExecution(types=types)
        assert sorted(
            type.value for type in unexpected_code_execution.types
        ) == sorted(types)

    def test_unexpected_code_execution_all_types_invalid(self):
        types = [
            "unauthorized_code_execution",
            "shell_command_execution",
            "eval_usage",
            "invalid",
        ]
        with pytest.raises(ValueError):
            UnexpectedCodeExecution(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        unexpected_code_execution = UnexpectedCodeExecution(
            types=["unauthorized_code_execution"]
        )
        test_cases = unexpected_code_execution.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(
            tc.vulnerability == "Unexpected Code Execution" for tc in test_cases
        )
        assert all(
            tc.vulnerability_type
            == UnexpectedCodeExecutionType.UNAUTHORIZED_CODE_EXECUTION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        unexpected_code_execution = UnexpectedCodeExecution(
            types=["shell_command_execution"], async_mode=False
        )

        def dummy_model_callback(prompt):
            return prompt

        results = unexpected_code_execution.assess(
            model_callback=dummy_model_callback,
        )

        assert unexpected_code_execution.is_vulnerable() is not None
        assert (
            unexpected_code_execution.simulated_attacks is not None
            and isinstance(unexpected_code_execution.simulated_attacks, dict)
        )
        assert unexpected_code_execution.res is not None and isinstance(
            unexpected_code_execution.res, dict
        )
        assert UnexpectedCodeExecutionType.SHELL_COMMAND_EXECUTION in results
        assert (
            len(results[UnexpectedCodeExecutionType.SHELL_COMMAND_EXECUTION])
            == 1
        )
        test_case = results[
            UnexpectedCodeExecutionType.SHELL_COMMAND_EXECUTION
        ][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_unexpected_code_execution_metric(self):
        from deepteam.metrics import UnexpectedCodeExecutionMetric

        unexpected_code_execution = UnexpectedCodeExecution(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = unexpected_code_execution._get_metric(
            UnexpectedCodeExecutionType.SHELL_COMMAND_EXECUTION
        )
        assert isinstance(metric, UnexpectedCodeExecutionMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        unexpected_code_execution = UnexpectedCodeExecution(
            types=["eval_usage"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await unexpected_code_execution.a_assess(
            model_callback=dummy_model_callback,
        )

        assert unexpected_code_execution.is_vulnerable() is not None
        assert (
            unexpected_code_execution.simulated_attacks is not None
            and isinstance(unexpected_code_execution.simulated_attacks, dict)
        )
        assert unexpected_code_execution.res is not None and isinstance(
            unexpected_code_execution.res, dict
        )
        assert UnexpectedCodeExecutionType.EVAL_USAGE in results
        assert len(results[UnexpectedCodeExecutionType.EVAL_USAGE]) == 1
        test_case = results[UnexpectedCodeExecutionType.EVAL_USAGE][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
