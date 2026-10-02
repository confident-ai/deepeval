"""
Security regression tests for the HumanEval code-execution opt-in gate.

Scoring HumanEval functional correctness executes untrusted, model-generated
code on the host. The restricted-builtins mapping is not a security boundary, so
execution must be disabled unless the operator explicitly opts in via
``DEEPEVAL_ALLOW_CODE_EXECUTION`` (and sandboxes the run).
"""

import pytest

from deepeval.benchmarks.human_eval.human_eval import (
    _execute_untrusted_code,
    ALLOW_CODE_EXECUTION_ENV_VAR,
)


class TestCodeExecutionGate:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv(ALLOW_CODE_EXECUTION_ENV_VAR, raising=False)
        with pytest.raises(RuntimeError, match="disabled by default"):
            _execute_untrusted_code("x = 1")

    def test_explicit_falsy_value_still_blocks(self, monkeypatch):
        monkeypatch.setenv(ALLOW_CODE_EXECUTION_ENV_VAR, "0")
        with pytest.raises(RuntimeError):
            _execute_untrusted_code("x = 1")

    def test_opt_in_allows_execution(self, monkeypatch):
        monkeypatch.setenv(ALLOW_CODE_EXECUTION_ENV_VAR, "1")
        result = _execute_untrusted_code("result = 6 * 7", local_vars={})
        assert result["result"] == 42

    def test_no_execution_side_effect_when_blocked(self, monkeypatch):
        """The blocked path must raise before running any of the code."""
        monkeypatch.delenv(ALLOW_CODE_EXECUTION_ENV_VAR, raising=False)
        local_vars = {}
        with pytest.raises(RuntimeError):
            _execute_untrusted_code("result = 123", local_vars=local_vars)
        assert "result" not in local_vars
