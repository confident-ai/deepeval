import pytest
import os
from deepeval.telemetry import telemetry_opt_out


def test_telemetry_opt_in_env_var():
    os.environ["DEEPEVAL_ENABLE_TELEMETRY"] = "YES"
    assert telemetry_opt_out() is False, "should be opted in"


@pytest.mark.parametrize("env_var_value", ["True", "true", "1", "yes"])
def test_telemetry_opt_out_env_var_set_to_not_yes(env_var_value):
    os.environ["DEEPEVAL_ENABLE_TELEMETRY"] = env_var_value
    assert telemetry_opt_out() is True, "should be opted out"


def test_telemetry_opt_out_env_var_not_set():
    del os.environ["DEEPEVAL_ENABLE_TELEMETRY"]
    assert telemetry_opt_out() is True, "should be opted out"
