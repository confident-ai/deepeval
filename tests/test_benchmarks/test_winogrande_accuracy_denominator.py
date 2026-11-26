import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from deepeval.benchmarks.winogrande import winogrande


@pytest.mark.parametrize(
    ("answers", "limit", "expected_scores", "expected_accuracy"),
    [
        (["1"], 2, [True], 1.0),
        (["1", "2"], 3, [True, False], 0.5),
        (["2"], 2, [False], 0.0),
        (["1", "2"], 2, [True, False], 0.5),
        (["1", "2"], 1, [True], 1.0),
        ([], 2, [], 0.0),
    ],
)
def test_winogrande_accuracy_uses_scored_rows(
    monkeypatch, answers, limit, expected_scores, expected_accuracy
):
    dataset_module = ModuleType("datasets")
    dataset_module.Dataset = object

    def reject_external_dataset(*args, **kwargs):
        pytest.fail("Winogrande should use its supplied validation rows")

    dataset_module.load_dataset = reject_external_dataset
    monkeypatch.setitem(sys.modules, "datasets", dataset_module)
    monkeypatch.setattr(
        winogrande, "capture_benchmark_run", lambda *args: nullcontext()
    )
    prompts = []

    def generate(*, prompt, schema):
        prompts.append(prompt)
        return SimpleNamespace(answer="A")

    rows = [
        {
            "sentence": f"Question {index}: _",
            "option1": "first",
            "option2": "second",
            "answer": answer,
        }
        for index, answer in enumerate(answers)
    ]
    benchmark = winogrande.Winogrande(
        n_shots=0, n_problems=limit, dataset={"validation": rows}
    )

    result = benchmark.evaluate(SimpleNamespace(generate=generate))

    assert result.overall_accuracy == expected_accuracy
    assert benchmark.overall_score == expected_accuracy
    assert benchmark.predictions["Correct"].tolist() == expected_scores
    assert len(prompts) == len(expected_scores)
