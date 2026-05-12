import pytest
import os
import yaml
from unittest.mock import patch, MagicMock
from deepeval.cli.cicd.cicd import (
    _build_goldens_from_dataset,
    _build_metrics,
    _load_config,
    _load_model_callback_from_file,
    _post_github_pr_comment,
    _post_ci_comment,
)
from deepeval.dataset import Golden, ConversationalGolden
from deepeval.metrics import AnswerRelevancyMetric

def test_build_goldens_from_dataset_single_turn():
    raw_dataset = {
        "goldens": [
            {"input": "What is the capital of France?", "expected_output": "Paris"}
        ]
    }
    goldens = _build_goldens_from_dataset(raw_dataset)
    assert len(goldens) == 1
    assert isinstance(goldens[0], Golden)
    assert goldens[0].input == "What is the capital of France?"
    assert goldens[0].expected_output == "Paris"

def test_build_goldens_from_dataset_multi_turn():
    raw_dataset = {
        "goldens": [
            {"scenario": "User wants to buy a ticket", "expected_outcome": "Ticket purchased"}
        ]
    }
    goldens = _build_goldens_from_dataset(raw_dataset)
    assert len(goldens) == 1
    assert isinstance(goldens[0], ConversationalGolden)
    assert goldens[0].scenario == "User wants to buy a ticket"
    assert goldens[0].expected_outcome == "Ticket purchased"

def test_build_goldens_from_dataset_mixed_error():
    raw_dataset = {
        "goldens": [
            {"input": "Hello", "expected_output": "Hi"},
            {"scenario": "Testing", "expected_outcome": "Tested"}
        ]
    }
    with pytest.raises(RuntimeError, match="Mixed golden types are not allowed"):
        _build_goldens_from_dataset(raw_dataset)

def test_build_goldens_from_dataset_empty_error():
    raw_dataset = {"goldens": []}
    with pytest.raises(ValueError, match="The 'goldens' list is empty"):
        _build_goldens_from_dataset(raw_dataset)

def test_build_goldens_from_dataset_alias_error():
    raw_dataset = {"alias": "fake_non_existent_alias_12345"}
    with pytest.raises(RuntimeError, match="Failed to pull dataset with alias"):
        _build_goldens_from_dataset(raw_dataset)

@patch("deepeval.cli.cicd.cicd.EvaluationDataset")
def test_build_goldens_from_dataset_alias_valid(mock_eval_dataset_class):
    # Setup the mock instance
    mock_instance = MagicMock()
    # Provide a fake golden in the dataset
    mock_instance.goldens = [Golden(input="test alias input", expected_output="test alias output")]
    mock_eval_dataset_class.return_value = mock_instance
    
    raw_dataset = {"alias": "valid_alias_name"}
    goldens = _build_goldens_from_dataset(raw_dataset)
    
    # Assertions
    mock_eval_dataset_class.assert_called_once()
    mock_instance.pull.assert_called_once_with(alias="valid_alias_name")
    assert len(goldens) == 1
    assert isinstance(goldens[0], Golden)
    assert goldens[0].input == "test alias input"
    assert goldens[0].expected_output == "test alias output"

def test_build_goldens_from_dataset_path_valid(tmp_path):
    import json
    json_file = tmp_path / "test_dataset.json"
    data = [{"input": "test input", "expected_output": "test output"}]
    with open(json_file, "w") as f:
        json.dump(data, f)
    
    raw_dataset = {"path": str(json_file)}
    goldens = _build_goldens_from_dataset(raw_dataset)
    assert len(goldens) == 1
    assert isinstance(goldens[0], Golden)
    assert goldens[0].input == "test input"
    assert goldens[0].expected_output == "test output"

def test_build_goldens_from_dataset_path_error():
    raw_dataset = {"path": "non_existent_file.json"}
    with pytest.raises(RuntimeError, match="Failed to load dataset from path"):
        _build_goldens_from_dataset(raw_dataset)

def test_build_metrics_valid():
    raw_metrics = [
        {"metric": "AnswerRelevancyMetric", "threshold": 0.6}
    ]
    metrics = _build_metrics(raw_metrics)
    assert len(metrics) == 1
    assert isinstance(metrics[0], AnswerRelevancyMetric)
    assert metrics[0].threshold == 0.6

def test_build_metrics_invalid_class():
    raw_metrics = [
        {"metric": "NonExistentMetric"}
    ]
    with pytest.raises(ValueError, match="Unknown metric 'NonExistentMetric'"):
        _build_metrics(raw_metrics)

def test_build_metrics_invalid_kwargs():
    raw_metrics = [
        {"metric": "AnswerRelevancyMetric", "invalid_arg": 123}
    ]
    with pytest.raises(TypeError, match="Failed to construct AnswerRelevancyMetric"):
        _build_metrics(raw_metrics)

def test_load_config_valid(tmp_path):
    config_data = {"dataset": {"goldens": [{"input": "test"}]}}
    config_file = tmp_path / "test_config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    loaded_config = _load_config(str(config_file))
    assert loaded_config == config_data

def test_load_config_invalid_path():
    with pytest.raises(RuntimeError, match="Could not read or parse YAML file"):
        _load_config("non_existent_file.yml")

def test_load_model_callback_valid(tmp_path):
    callback_file = tmp_path / "callback.py"
    with open(callback_file, "w") as f:
        f.write("def my_callback(input_str):\n    return 'response'")
    
    callback = _load_model_callback_from_file(str(callback_file), "my_callback")
    assert callable(callback)
    assert callback("test") == "response"

def test_load_model_callback_missing_file():
    with pytest.raises(FileNotFoundError, match="Model callback file not found"):
        _load_model_callback_from_file("non_existent.py", "my_callback")

def test_load_model_callback_missing_function(tmp_path):
    callback_file = tmp_path / "callback.py"
    with open(callback_file, "w") as f:
        f.write("def other_function():\n    pass")
    
    with pytest.raises(AttributeError, match="Function 'my_callback' not found"):
        _load_model_callback_from_file(str(callback_file), "my_callback")

@patch("deepeval.cli.cicd.cicd.os.environ.get")
@patch("deepeval.cli.cicd.cicd.requests.get")
@patch("deepeval.cli.cicd.cicd.requests.post")
def test_post_github_pr_comment_new(mock_post, mock_get, mock_env_get, tmp_path):
    # Setup mock environment
    mock_env_get.side_effect = lambda k: {
        "GITHUB_TOKEN": "fake_token",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_EVENT_PATH": str(tmp_path / "event.json")
    }.get(k)
    
    # Create fake event file
    with open(tmp_path / "event.json", "w") as f:
        f.write('{"pull_request": {"number": 123}}')
        
    # Mock GET response (no existing comments)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_get.return_value = mock_response
    
    # Mock POST response
    mock_post_response = MagicMock()
    mock_post_response.status_code = 201
    mock_post.return_value = mock_post_response
    
    _post_github_pr_comment("Test comment")
    
    mock_get.assert_called_once()
    mock_post.assert_called_once()
    assert mock_post.call_args[1]["json"]["body"] == "Test comment"

@patch("deepeval.cli.cicd.cicd.os.environ.get")
@patch("deepeval.cli.cicd.cicd.requests.get")
@patch("deepeval.cli.cicd.cicd.requests.patch")
def test_post_github_pr_comment_update(mock_patch, mock_get, mock_env_get, tmp_path):
    # Setup mock environment
    mock_env_get.side_effect = lambda k: {
        "GITHUB_TOKEN": "fake_token",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_EVENT_PATH": str(tmp_path / "event.json")
    }.get(k)
    
    # Create fake event file
    with open(tmp_path / "event.json", "w") as f:
        f.write('{"pull_request": {"number": 123}}')
        
    # Mock GET response (existing comment found)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"id": 456, "body": "🚀 DeepEval Evaluation Results previously"}]
    mock_get.return_value = mock_response
    
    # Mock PATCH response
    mock_patch_response = MagicMock()
    mock_patch_response.status_code = 200
    mock_patch.return_value = mock_patch_response
    
    _post_github_pr_comment("Updated comment")
    
    mock_get.assert_called_once()
    mock_patch.assert_called_once()
    assert mock_patch.call_args[1]["json"]["body"] == "Updated comment"
    assert "456" in mock_patch.call_args[0][0]

@patch("deepeval.cli.cicd.cicd._post_github_pr_comment")
def test_post_ci_comment_github(mock_github):
    _post_ci_comment("Test", "github")
    mock_github.assert_called_once_with("Test")

@patch("deepeval.cli.cicd.cicd.print")
def test_post_ci_comment_unsupported(mock_print):
    _post_ci_comment("Test", "gitlab")
    mock_print.assert_called_once()
    assert "not yet supported" in mock_print.call_args[0][0]
