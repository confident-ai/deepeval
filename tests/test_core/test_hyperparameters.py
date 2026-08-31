import pytest

from deepeval.test_run import (
    global_test_run_manager,
    log_hyperparameters,
)


@pytest.fixture(autouse=True)
def fresh_test_run():
    global_test_run_manager.reset()
    global_test_run_manager.create_test_run()
    yield
    global_test_run_manager.reset()


def test_log_hyperparameters_bare_form_sets_hyperparameters():
    @log_hyperparameters
    def run():
        return {"model": "gpt-4", "temperature": 0.7}

    assert run() == {"model": "gpt-4", "temperature": 0.7}
    assert global_test_run_manager.get_test_run().hyperparameters == {
        "model": "gpt-4",
        "temperature": "0.7",
    }


def test_log_hyperparameters_called_form():
    @log_hyperparameters()
    def run():
        return {"max_tokens": 128}

    assert run() == {"max_tokens": 128}
    assert global_test_run_manager.get_test_run().hyperparameters == {
        "max_tokens": "128",
    }


def test_log_hyperparameters_with_kwargs_merges_returned_dict():
    @log_hyperparameters(model="gpt-4")
    def run():
        return {"temperature": 0.2}

    assert run() == {"temperature": 0.2}
    hyperparameters = global_test_run_manager.get_test_run().hyperparameters
    assert hyperparameters == {"model": "gpt-4", "temperature": "0.2"}


def test_log_hyperparameters_returned_dict_takes_precedence():
    @log_hyperparameters(model="gpt-4")
    def run():
        return {"model": "gpt-4o"}

    assert run() == {"model": "gpt-4o"}
    hyperparameters = global_test_run_manager.get_test_run().hyperparameters
    assert hyperparameters == {"model": "gpt-4o"}


def test_log_hyperparameters_none_return_yields_none_hyperparameters():
    @log_hyperparameters
    def run():
        return None

    assert run() is None
    assert global_test_run_manager.get_test_run().hyperparameters is None


def test_log_hyperparameters_non_dict_return_raises_type_error():
    with pytest.raises(TypeError, match="Hyperparameters must be a dictionary"):

        @log_hyperparameters
        def run():
            return "not a dict"


def test_log_hyperparameters_wrapper_preserves_callable_result():
    @log_hyperparameters(seed=42)
    def run():
        return {"temperature": 0.1}

    # The decorated function stays callable and returns its original value.
    assert run() == {"temperature": 0.1}
    assert global_test_run_manager.get_test_run().hyperparameters == {
        "seed": "42",
        "temperature": "0.1",
    }


def test_log_hyperparameters_skips_none_values():
    @log_hyperparameters(model=None)
    def run():
        return {"temperature": 0.5}

    assert run() == {"temperature": 0.5}
    assert global_test_run_manager.get_test_run().hyperparameters == {
        "temperature": "0.5",
    }
