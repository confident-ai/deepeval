import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from deepeval.scorer import Scorer


@pytest.fixture
def bert_dependencies(monkeypatch):
    bert_score = ModuleType("bert_score")
    bert_score.BERTScorer = Mock()
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "bert_score", bert_score)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return bert_score.BERTScorer


@pytest.mark.parametrize(
    "module,package", [("bert_score", "bert-score"), ("torch", "torch")]
)
@pytest.mark.parametrize("error_type", [ModuleNotFoundError, ImportError])
def test_bert_score_dependency_error(
    monkeypatch, bert_dependencies, module, package, error_type
):
    original_import = builtins.__import__
    error = error_type(f"Cannot import {module}")

    def import_without_dependency(name, *args, **kwargs):
        if name == module:
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dependency)

    with pytest.raises(ImportError, match=f"pip install {package}") as exc:
        Scorer.bert_score("reference", "prediction")

    assert exc.value.__cause__ is error
    bert_dependencies.assert_not_called()


def test_bert_score_with_dependencies(bert_dependencies):
    tensors = [Mock(), Mock(), Mock()]
    for tensor, value in zip(tensors, [0.8, 0.7, 0.75]):
        tensor.detach.return_value.numpy.return_value.tolist.return_value = [
            value
        ]
    bert_dependencies.return_value.score.return_value = tensors

    result = Scorer.bert_score("reference", "prediction")

    assert result == {
        "bert-precision": [0.8],
        "bert-recall": [0.7],
        "bert-f1": [0.75],
    }
    bert_dependencies.return_value.score.assert_called_once_with(
        cands=["prediction"], refs=["reference"]
    )
