import sys
import types
import importlib.util

import pytest

from deepeval.models.detoxify_model import DetoxifyModel


def test_importing_module_does_not_require_torch_or_detoxify():
    """
    Regression test for issue #382: before the fix, `detoxify_model.py` did
    hard top-level `import torch` / `from detoxify import Detoxify`, so this
    import alone crashed with ModuleNotFoundError on a stock install.
    """
    assert DetoxifyModel is not None


def test_constructing_without_detoxify_raises_clean_import_error():
    if importlib.util.find_spec("detoxify") is not None:
        pytest.skip("detoxify is installed in this environment")

    with pytest.raises(ImportError, match=r"pip install deepeval\[toxicity\]"):
        DetoxifyModel()


def test_load_model_returns_detoxify_instance_when_available(monkeypatch):
    class FakeDetoxify:
        def __init__(self, model_name, device):
            self.model_name = model_name
            self.device = device

        def predict(self, text):
            return {"toxicity": 0.1, "severe_toxicity": 0.0}

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_detoxify_module = types.SimpleNamespace(Detoxify=FakeDetoxify)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "detoxify", fake_detoxify_module)

    model = DetoxifyModel()

    assert isinstance(model.model, FakeDetoxify)
    assert model.model.device == "cpu"

    score, breakdown = model._call("some text")

    assert score == pytest.approx(0.05)
    assert breakdown == {"toxicity": 0.1, "severe_toxicity": 0.0}
