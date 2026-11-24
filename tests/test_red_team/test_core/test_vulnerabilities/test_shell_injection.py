import pytest

from deepteam.vulnerabilities import ShellInjection
from deepteam.vulnerabilities.shell_injection import ShellInjectionType
from deepteam.test_case import RTTestCase


class TestShellInjection:

    def test_shell_injection_all_types(self):
        types = [
            "command_injection",
            "system_command_execution",
            "shell_escape_sequences",
        ]
        shell_injection = ShellInjection(types=types)
        assert sorted(type.value for type in shell_injection.types) == sorted(
            types
        )

    def test_shell_injection_all_types_default(self):
        shell_injection = ShellInjection()
        assert sorted(type.value for type in shell_injection.types) == sorted(
            type.value for type in ShellInjectionType
        )

    def test_shell_injection_command_injection(self):
        types = ["command_injection"]
        shell_injection = ShellInjection(types=types)
        assert sorted(type.value for type in shell_injection.types) == sorted(
            types
        )

    def test_shell_injection_system_command_execution(self):
        types = ["system_command_execution"]
        shell_injection = ShellInjection(types=types)
        assert sorted(type.value for type in shell_injection.types) == sorted(
            types
        )

    def test_shell_injection_shell_escape_sequences(self):
        types = ["shell_escape_sequences"]
        shell_injection = ShellInjection(types=types)
        assert sorted(type.value for type in shell_injection.types) == sorted(
            types
        )

    def test_shell_injection_all_types_invalid(self):
        types = [
            "command_injection",
            "system_command_execution",
            "shell_escape_sequences",
            "invalid",
        ]
        with pytest.raises(ValueError):
            ShellInjection(types=types)

    def test_simulate_attacks_returns_expected_cases(self):
        shell_injection = ShellInjection(types=["command_injection"])
        test_cases = shell_injection.simulate_attacks(
            attacks_per_vulnerability_type=2
        )

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "Shell Injection" for tc in test_cases)
        assert all(
            tc.vulnerability_type == ShellInjectionType.COMMAND_INJECTION
            for tc in test_cases
        )

    def test_assess_returns_results(self):
        shell_injection = ShellInjection(
            types=["command_injection"], async_mode=False
        )

        def dummy_model_callback(prompt):
            # Provide a simple pass-through or minimal callback if required by your real env
            return prompt

        results = shell_injection.assess(
            model_callback=dummy_model_callback,
        )

        assert shell_injection.is_vulnerable() is not None
        assert shell_injection.simulated_attacks is not None and isinstance(
            shell_injection.simulated_attacks, dict
        )
        assert shell_injection.res is not None and isinstance(
            shell_injection.res, dict
        )
        assert ShellInjectionType.COMMAND_INJECTION in results
        assert len(results[ShellInjectionType.COMMAND_INJECTION]) == 1
        test_case = results[ShellInjectionType.COMMAND_INJECTION][0]
        # The score and reason will depend on the real metric evaluation
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_ShellInjection_metric(self):
        from deepteam.metrics import ShellInjectionMetric

        shell_injection = ShellInjection(
            async_mode=True, verbose_mode=True, evaluation_model="gpt-4o"
        )
        metric = shell_injection._get_metric(
            ShellInjectionType.COMMAND_INJECTION
        )
        assert isinstance(metric, ShellInjectionMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        shell_injection = ShellInjection(
            types=["command_injection"], async_mode=True
        )

        async def dummy_model_callback(prompt):
            return prompt

        results = await shell_injection.a_assess(
            model_callback=dummy_model_callback,
        )

        assert shell_injection.is_vulnerable() is not None
        assert shell_injection.simulated_attacks is not None and isinstance(
            shell_injection.simulated_attacks, dict
        )
        assert shell_injection.res is not None and isinstance(
            shell_injection.res, dict
        )
        assert ShellInjectionType.COMMAND_INJECTION in results
        assert len(results[ShellInjectionType.COMMAND_INJECTION]) == 1
        test_case = results[ShellInjectionType.COMMAND_INJECTION][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
