import importlib
import sys
from types import ModuleType, SimpleNamespace


def _load_summac_model(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.nn = SimpleNamespace(Module=object)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "nltk", ModuleType("nltk"))
    monkeypatch.delitem(
        sys.modules, "deepeval.models._summac_model", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "deepeval.models.summac_model", raising=False
    )
    return importlib.import_module("deepeval.models.summac_model")


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def score(self, targets, predictions):
        self.calls.append(("score", targets, predictions))
        return {"scores": [0.75]}

    def score_one(self, target, prediction):
        self.calls.append(("score_one", target, prediction))
        return {"score": 0.5}


def test_summac_dispatches_list_inputs_to_batch_score(monkeypatch):
    summac_model = _load_summac_model(monkeypatch)
    model = object.__new__(summac_model.SummaCModels)
    model.model = _RecordingBackend()

    result = model._call(["prediction"], ["target"])

    assert result == {"scores": [0.75]}
    assert model.model.calls == [
        ("score", ["target"], ["prediction"]),
    ]


def test_summac_dispatches_string_inputs_to_single_score(monkeypatch):
    summac_model = _load_summac_model(monkeypatch)
    model = object.__new__(summac_model.SummaCModels)
    model.model = _RecordingBackend()

    result = model._call("prediction", "target")

    assert result == {"score": 0.5}
    assert model.model.calls == [
        ("score_one", "target", "prediction"),
    ]
