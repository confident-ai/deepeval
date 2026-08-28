"""Documented @log_hyperparameters decorator forms must not raise TypeError."""

import pytest

import deepeval
from deepeval.test_run import global_test_run_manager


@pytest.fixture(autouse=True)
def isolated_test_run():
    global_test_run_manager.reset()
    global_test_run_manager.create_test_run(identifier="log-hyperparameters")
    yield
    global_test_run_manager.reset()


def test_log_hyperparameters_empty_call_logs_returned_dict():
    @deepeval.log_hyperparameters()
    def hyperparameters():
        return {"temperature": 1}

    test_run = global_test_run_manager.get_test_run()
    assert test_run.hyperparameters == {"temperature": "1"}


def test_log_hyperparameters_keyword_form_merges_decorator_kwargs():
    @deepeval.log_hyperparameters(model="gpt-4", prompt_template="...")
    def hyperparameters():
        return {"temperature": 1, "chunk size": 500}

    test_run = global_test_run_manager.get_test_run()
    assert test_run.hyperparameters == {
        "model": "gpt-4",
        "prompt_template": "...",
        "temperature": "1",
        "chunk size": "500",
    }


def test_log_hyperparameters_bare_form_still_works():
    @deepeval.log_hyperparameters
    def hyperparameters():
        return {"temperature": 1}

    test_run = global_test_run_manager.get_test_run()
    assert test_run.hyperparameters == {"temperature": "1"}
