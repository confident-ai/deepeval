"""Offline tests for HallucinationModel / Scorer.hallucination_score kwargs.

CrossEncoder is mocked so these tests never download Hugging Face models.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from deepeval.models.hallucination_model import (
    DEFAULT_HALLUCINATION_MODEL,
    HallucinationModel,
)
from deepeval.scorer.scorer import Scorer
from deepeval.singleton import Singleton


@pytest.fixture(autouse=True)
def _clear_singleton_cache():
    Singleton._instances.clear()
    yield
    Singleton._instances.clear()


@pytest.fixture
def mock_cross_encoder():
    mock_cls = MagicMock(name="CrossEncoder")
    mock_instance = MagicMock()
    mock_instance.predict.return_value = 0.42
    mock_cls.return_value = mock_instance

    mock_module = MagicMock()
    mock_module.CrossEncoder = mock_cls
    with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
        yield mock_cls


def test_default_model_trusts_remote_code(mock_cross_encoder):
    HallucinationModel()
    mock_cross_encoder.assert_called_once_with(
        DEFAULT_HALLUCINATION_MODEL, trust_remote_code=True
    )


def test_explicit_vectara_model_trusts_remote_code(mock_cross_encoder):
    HallucinationModel(model_name=DEFAULT_HALLUCINATION_MODEL)
    mock_cross_encoder.assert_called_once_with(
        DEFAULT_HALLUCINATION_MODEL, trust_remote_code=True
    )


def test_custom_model_does_not_trust_remote_code_by_default(mock_cross_encoder):
    HallucinationModel(model_name="custom/hallucination-model")
    mock_cross_encoder.assert_called_once_with(
        "custom/hallucination-model", trust_remote_code=False
    )


def test_explicit_trust_remote_code_false_on_default_model(mock_cross_encoder):
    HallucinationModel(trust_remote_code=False)
    mock_cross_encoder.assert_called_once_with(
        DEFAULT_HALLUCINATION_MODEL, trust_remote_code=False
    )


def test_explicit_trust_remote_code_true_on_custom_model(mock_cross_encoder):
    HallucinationModel(
        model_name="custom/hallucination-model", trust_remote_code=True
    )
    mock_cross_encoder.assert_called_once_with(
        "custom/hallucination-model", trust_remote_code=True
    )


def test_extra_cross_encoder_kwargs_are_forwarded(mock_cross_encoder):
    HallucinationModel(device="cpu", max_length=256)
    mock_cross_encoder.assert_called_once_with(
        DEFAULT_HALLUCINATION_MODEL,
        trust_remote_code=True,
        device="cpu",
        max_length=256,
    )


def test_scorer_forwards_trust_remote_code_and_kwargs():
    mock_instance = MagicMock()
    mock_instance.model.predict.return_value = 0.5

    with patch(
        "deepeval.models.hallucination_model.HallucinationModel",
        return_value=mock_instance,
    ) as mock_hm:
        score = Scorer.hallucination_score(
            "source text",
            "prediction text",
            model="custom/hallucination-model",
            trust_remote_code=True,
            device="cpu",
        )

    mock_hm.assert_called_once_with(
        model_name="custom/hallucination-model",
        trust_remote_code=True,
        device="cpu",
    )
    mock_instance.model.predict.assert_called_once_with(
        ["source text", "prediction text"]
    )
    assert score == 0.5


def test_scorer_default_model_omits_explicit_flag():
    mock_instance = MagicMock()
    mock_instance.model.predict.return_value = 0.1

    with patch(
        "deepeval.models.hallucination_model.HallucinationModel",
        return_value=mock_instance,
    ) as mock_hm:
        Scorer.hallucination_score("source", "prediction")

    mock_hm.assert_called_once_with(model_name=None, trust_remote_code=None)
