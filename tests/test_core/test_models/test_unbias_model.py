import sys
import types
import importlib.util

import pytest

from deepeval.models.unbias_model import UnBiasedModel


def test_constructing_without_dbias_raises_clean_import_error():
    """
    Regression test for issue #382: Dbias is not a declared dependency, so
    on a stock install `UnBiasedModel()` must fail with a clear ImportError
    telling the user how to install it, not an UnboundLocalError.
    """
    if importlib.util.find_spec("Dbias") is not None:
        pytest.skip("Dbias is installed in this environment")

    with pytest.raises(ImportError, match=r"pip install deepeval\[bias\]"):
        UnBiasedModel()


def test_load_model_returns_classifier_when_dbias_available(monkeypatch):
    fake_classifier = lambda text: [{"label": "Biased", "score": 0.91}]
    fake_dbias = types.ModuleType("Dbias")
    fake_bias_classification = types.SimpleNamespace(classifier=fake_classifier)
    monkeypatch.setitem(sys.modules, "Dbias", fake_dbias)
    monkeypatch.setitem(
        sys.modules, "Dbias.bias_classification", fake_bias_classification
    )

    model = UnBiasedModel()

    assert model.model is fake_classifier
    assert model._call("some text") == fake_classifier("some text")
