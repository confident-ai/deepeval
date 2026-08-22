import sys
from types import ModuleType

from deepeval.benchmarks.schema import BinaryChoiceSchema
from deepeval.benchmarks.winogrande.winogrande import Winogrande
from deepeval.dataset import Golden


class _SchemaUnsupportedModel:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, schema=None):
        self.calls.append((prompt, schema))
        if schema is not None:
            raise TypeError("schema generation is not supported")
        return "A"


def test_fallback_prompt_only_allows_winogrande_answers(monkeypatch):
    datasets = ModuleType("datasets")
    datasets.Dataset = object
    pandas = ModuleType("pandas")
    pandas.DataFrame = object
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "pandas", pandas)

    benchmark = Winogrande(n_shots=0, n_problems=1)
    model = _SchemaUnsupportedModel()
    golden = Golden(
        input="Sentence: A _ ran.\nA. dog\nB. cat\nAnswer:", expected_output="A"
    )

    result = benchmark.predict(model, golden)

    assert model.calls[0][1] is BinaryChoiceSchema
    assert model.calls[1][1] is None
    assert model.calls[1][0].endswith(
        "\n\nOutput 'A' or 'B'. Full answer not needed."
    )
    assert result == {"prediction": "A", "score": 1}
