import pytest
import os
from deepteam.cli.main import run, _load_config, _load_callback_from_file
from deepteam.cli.model_callback import load_model
from deepteam.red_teamer.risk_assessment import RiskAssessment
from deepeval.models import DeepEvalBaseLLM


def test_cli_run_return_both_risk_and_file(tmp_path):
    OUTPUT_FOLDER = tmp_path / "results_file"

    test_dir = os.path.dirname(__file__)
    config_path = os.path.join(test_dir, "config.yaml")
    result = run(config_path, 1, 1, str(OUTPUT_FOLDER))

    assert result.risk_assessment is not None
    assert result.file_path is not None
    assert isinstance(result.risk_assessment, RiskAssessment)
    assert str(result.file_path).startswith(str(OUTPUT_FOLDER))


def test_cli_run_return_both_risk_and_none():
    test_dir = os.path.dirname(__file__)
    config_path = os.path.join(test_dir, "config.yaml")
    result = run(config_path, 1, 1, None)

    assert result.risk_assessment is not None
    assert result.file_path is None
    assert isinstance(result.risk_assessment, RiskAssessment)


def test_target_loads_from_yaml_callback():
    TEST = "Check callback returns same @123"

    test_dir = os.path.dirname(__file__)
    config_path = os.path.join(test_dir, "config_for_callback.yaml")
    cfg = _load_config(config_path)
    target_cfg = cfg.get("target", {})

    callback_cfg = target_cfg["callback"]
    file_path = callback_cfg.get("file")
    function_name = callback_cfg.get("function")

    model_callback = _load_callback_from_file(file_path, function_name)

    assert callable(model_callback)
    assert model_callback(TEST) == TEST


def test_models_loads_from_yaml_class():

    test_dir = os.path.dirname(__file__)
    config_path = os.path.join(test_dir, "config_for_class.yaml")
    cfg = _load_config(config_path)
    models_cfg = cfg.get("models")

    simulator_model_spec = models_cfg.get("simulator")
    _simulator_model_spec = simulator_model_spec["model"]
    simulator_model = load_model(_simulator_model_spec)

    evaluation_model_spec = models_cfg.get("evaluation")
    _evaluation_model_spec = evaluation_model_spec["model"]
    evaluation_model = load_model(_evaluation_model_spec)

    assert models_cfg is not None
    assert (
        simulator_model_spec is not None and evaluation_model_spec is not None
    )
    assert (
        _simulator_model_spec is not None and _evaluation_model_spec is not None
    )
    assert isinstance(simulator_model, DeepEvalBaseLLM)
    assert isinstance(evaluation_model, DeepEvalBaseLLM)
