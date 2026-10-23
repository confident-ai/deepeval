import pytest
from unittest.mock import patch, MagicMock
from deepeval.test_run.hyperparameters import log_hyperparameters


@pytest.fixture
def mock_test_run():
    with patch(
        "deepeval.test_run.hyperparameters.global_test_run_manager"
    ) as mock_mgr:
        test_run = MagicMock()
        test_run.hyperparameters = None
        mock_mgr.get_test_run.return_value = test_run
        yield test_run, mock_mgr


def test_log_hyperparameters_bare_decorator(mock_test_run):
    test_run, mock_mgr = mock_test_run

    @log_hyperparameters
    def my_hyperparameters():
        """Docstring for my_hyperparameters."""
        return {"model": "gpt-4o", "temperature": 0.7}

    assert my_hyperparameters.__name__ == "my_hyperparameters"
    assert my_hyperparameters.__doc__ == "Docstring for my_hyperparameters."
    assert test_run.hyperparameters == {"model": "gpt-4o", "temperature": "0.7"}
    mock_mgr.save_test_run.assert_called_once()
    assert my_hyperparameters() == {"model": "gpt-4o", "temperature": 0.7}


def test_log_hyperparameters_empty_parentheses(mock_test_run):
    test_run, mock_mgr = mock_test_run

    @log_hyperparameters()
    def my_hyperparameters():
        return {"model": "gpt-4.1", "temperature": 1.0}

    assert test_run.hyperparameters == {
        "model": "gpt-4.1",
        "temperature": "1.0",
    }
    mock_mgr.save_test_run.assert_called_once()


def test_log_hyperparameters_with_kwargs(mock_test_run):
    test_run, mock_mgr = mock_test_run

    @log_hyperparameters(model="gpt-4", prompt_template="test-template")
    def my_hyperparameters():
        return {"temperature": 0.5}

    assert test_run.hyperparameters == {
        "model": "gpt-4",
        "prompt_template": "test-template",
        "temperature": "0.5",
    }
    mock_mgr.save_test_run.assert_called_once()


def test_log_hyperparameters_with_kwargs_and_empty_func(mock_test_run):
    test_run, mock_mgr = mock_test_run

    @log_hyperparameters(model="gpt-4o-mini", chunk_size=512)
    def my_hyperparameters():
        pass

    assert test_run.hyperparameters == {
        "model": "gpt-4o-mini",
        "chunk_size": "512",
    }
    mock_mgr.save_test_run.assert_called_once()


def test_log_hyperparameters_invalid_return_type(mock_test_run):
    with pytest.raises(
        TypeError,
        match="Hyperparameters function must return a dictionary or None",
    ):

        @log_hyperparameters
        def my_hyperparameters():
            return "not-a-dict"


def test_log_hyperparameters_invalid_decorator_usage():
    with pytest.raises(
        TypeError, match="log_hyperparameters can only be used as a decorator"
    ):
        log_hyperparameters("not_callable", "extra_arg")
