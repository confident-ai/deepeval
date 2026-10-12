from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import deepeval
import deepeval.test_run.hyperparameters as hyperparameters_module
from deepeval.test_run.test_run import TEMP_FILE_PATH


@pytest.fixture
def stub_test_run_manager(monkeypatch):
    test_run = SimpleNamespace(hyperparameters=None)
    manager = SimpleNamespace(
        get_test_run=Mock(return_value=test_run),
        save_test_run=Mock(),
    )
    monkeypatch.setattr(
        hyperparameters_module,
        "global_test_run_manager",
        manager,
    )
    return test_run, manager


def test_log_hyperparameters_supports_bare_decorator(stub_test_run_manager):
    test_run, manager = stub_test_run_manager

    @hyperparameters_module.log_hyperparameters
    def hyperparameters(value="registered"):
        return {"value": value}

    assert test_run.hyperparameters == {"value": "registered"}
    assert hyperparameters.__name__ == "hyperparameters"
    assert hyperparameters("called") == {"value": "called"}
    manager.save_test_run.assert_called_once_with(TEMP_FILE_PATH)


def test_log_hyperparameters_supports_empty_call(stub_test_run_manager):
    test_run, _ = stub_test_run_manager

    @deepeval.log_hyperparameters()
    def hyperparameters():
        return {"chunk size": 500}

    assert test_run.hyperparameters == {"chunk size": "500"}


def test_log_hyperparameters_merges_decorator_parameters(
    stub_test_run_manager,
):
    test_run, _ = stub_test_run_manager

    @deepeval.log_hyperparameters(
        model="gpt-4",
        prompt_template="new prompt",
    )
    def hyperparameters():
        return {
            "model": "old-model",
            "prompt template": "old prompt",
            "temperature": 1,
        }

    assert test_run.hyperparameters == {
        "model": "gpt-4",
        "prompt template": "new prompt",
        "temperature": "1",
    }


def test_log_hyperparameters_supports_model_without_prompt_template(
    stub_test_run_manager,
):
    test_run, _ = stub_test_run_manager

    @hyperparameters_module.log_hyperparameters(model="gpt-4")
    def hyperparameters():
        return {"temperature": 0}

    assert test_run.hyperparameters == {
        "model": "gpt-4",
        "temperature": "0",
    }


def test_log_hyperparameters_preserves_none(stub_test_run_manager):
    test_run, _ = stub_test_run_manager

    @hyperparameters_module.log_hyperparameters()
    def hyperparameters():
        return None

    assert test_run.hyperparameters is None


def test_log_hyperparameters_merges_parameters_when_return_is_none(
    stub_test_run_manager,
):
    test_run, _ = stub_test_run_manager

    @hyperparameters_module.log_hyperparameters(model="gpt-4")
    def hyperparameters():
        return None

    assert test_run.hyperparameters == {"model": "gpt-4"}


def test_log_hyperparameters_rejects_non_dict_return(
    stub_test_run_manager,
):
    _, manager = stub_test_run_manager

    with pytest.raises(
        TypeError,
        match="Hyperparameters must be a dictionary or None",
    ):

        @hyperparameters_module.log_hyperparameters()
        def hyperparameters():
            return []

    manager.save_test_run.assert_not_called()
